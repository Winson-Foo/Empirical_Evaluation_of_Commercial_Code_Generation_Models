/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ml.inference.nlp;

import org.elasticsearch.ElasticsearchStatusException;
import org.elasticsearch.common.logging.LoggerMessageFormat;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.xpack.core.ml.inference.results.InferenceResults;
import org.elasticsearch.xpack.core.ml.inference.results.NlpClassificationInferenceResults;
import org.elasticsearch.xpack.core.ml.inference.results.TopClassEntry;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.NlpConfig;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.Tokenization;
import org.elasticsearch.xpack.core.ml.inference.trainedmodel.ZeroShotClassificationConfig;
import org.elasticsearch.xpack.core.ml.utils.ExceptionsHelper;
import org.elasticsearch.xpack.ml.inference.nlp.tokenizers.NlpTokenizer;
import org.elasticsearch.xpack.ml.inference.nlp.tokenizers.TokenizationResult;
import org.elasticsearch.xpack.ml.inference.pytorch.results.PyTorchInferenceResult;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.elasticsearch.xpack.core.ml.inference.trainedmodel.InferenceConfig.DEFAULT_RESULTS_FIELD;

public class ZeroShotClassificationProcessor extends NlpTask.Processor {

    private final int entailmentPos;
    private final int contraPos;
    private final String[] labels;
    private final String hypothesisTemplate;
    private final boolean isMultiLabel;
    private final String resultsField;

    private static final String ENTAILMENT = "entailment";
    private static final String CONTRADICTION = "contradiction";

    private static final int MIN_INNER_CLASSIFICATION_RESULT_LENGTH = 3;

    public ZeroShotClassificationProcessor(NlpTokenizer tokenizer, ZeroShotClassificationConfig config) {
        super(tokenizer);
        List<String> lowerCased = config.getClassificationLabels()
                                       .stream()
                                       .map(s -> s.toLowerCase(Locale.ROOT))
                                       .toList();
        this.entailmentPos = lowerCased.indexOf(ENTAILMENT);
        this.contraPos = lowerCased.indexOf(CONTRADICTION);

        if (entailmentPos == -1 || contraPos == -1) {
            throw ExceptionsHelper.badRequestException("zero_shot_classification requires [entailment] and [contradiction] in classification_labels");
        }

        this.labels = config.getLabels().orElse(List.of()).toArray(String[]::new);
        this.hypothesisTemplate = config.getHypothesisTemplate();
        this.isMultiLabel = config.isMultiLabel();
        this.resultsField = config.getResultsField();
    }

    @Override
    public void validateInputs(List<String> inputs) {
        // nothing to validate
    }

    @Override
    public NlpTask.RequestBuilder getRequestBuilder(NlpConfig nlpConfig) {
        final String[] labelsValue = getLabelsValue(nlpConfig);
        if (labelsValue.length == 0) {
            throw ExceptionsHelper.badRequestException("zero_shot_classification requires non-empty [labels]");
        }

        return new RequestBuilder(tokenizer, labelsValue, hypothesisTemplate);
    }

    @Override
    public NlpTask.ResultProcessor getResultProcessor(NlpConfig nlpConfig) {
        final String[] labelsValue = getLabelsValue(nlpConfig);

        return new ResultProcessor(entailmentPos, contraPos, labelsValue, isMultiLabel, resultsField);
    }

    private String[] getLabelsValue(NlpConfig nlpConfig) {
        return (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig)
                ? zeroShotConfig.getLabels().orElse(List.of()).toArray(new String[0])
                : this.labels;
    }

    record RequestBuilder(NlpTokenizer tokenizer, String[] labels, String hypothesisTemplate)
            implements NlpTask.RequestBuilder {

        @Override
        public NlpTask.Request buildRequest(List<String> inputs, String requestId, Tokenization.Truncate truncate, int span) throws IOException {
            if (inputs.size() > 1) {
                throw ExceptionsHelper.badRequestException("Unable to do zero-shot classification on more than one text input at a time");
            }
            if (span > -1) {
                throw ExceptionsHelper.badRequestException("Unable to span zero-shot classification on long text input");
            }

            List<TokenizationResult.Tokens> tokenizations = new ArrayList<>(labels.length);
            int seqId = 0;
            final NlpTokenizer.InnerTokenization firstSequenceTokenization = tokenizer.innerTokenize(inputs.get(0));

            for (String label : labels) {
                tokenizations.add(getTokenizationResultForLabel(inputs, firstSequenceTokenization, label, truncate, seqId++));
            }

            TokenizationResult result = tokenizer.buildTokenizationResult(tokenizations);
            return result.buildRequest(requestId, truncate);
        }

        private TokenizationResult.Tokens getTokenizationResultForLabel(List<String> inputs, NlpTokenizer.InnerTokenization firstSequenceTokenization,
                                                                                String label, Tokenization.Truncate truncate, int seqId) throws IOException {
            final String formattedHypothesis = LoggerMessageFormat.format(null, hypothesisTemplate, label);
            return tokenizer.tokenize(inputs.get(0), firstSequenceTokenization, formattedHypothesis, truncate, seqId);
        }
    }

    private record ResultEntry(String label, double score) {}
    private record Result(NlpClassificationInferenceResults results, double[] normalizedScores) {}

    record ResultProcessor(int entailmentPos, int contraPos, String[] labels, boolean isMultiLabel, String resultsField)
            implements NlpTask.ResultProcessor {

        private static final int MIN_RESULTS_LENGTH = 1;

        @Override
        public InferenceResults processResult(TokenizationResult tokenization, PyTorchInferenceResult pyTorchResult) {
            if (pyTorchResult.getInferenceResult().length < 1) {
                throw new ElasticsearchStatusException("Zero shot classification result has no data", RestStatus.INTERNAL_SERVER_ERROR);
            }
            if (pyTorchResult.getInferenceResult()[0].length < MIN_RESULTS_LENGTH) {
                throw new ElasticsearchStatusException("Zero shot classification result length should be at least 1", RestStatus.INTERNAL_SERVER_ERROR);
            }
            if (pyTorchResult.getInferenceResult()[0].length != labels.length) {
                throw new ElasticsearchStatusException(
                        "Expected exactly [{}] values in zero shot classification result; got [{}]",
                        RestStatus.INTERNAL_SERVER_ERROR,
                        labels.length,
                        pyTorchResult.getInferenceResult().length
                );
            }

            Result result = getResult(pyTorchResult, labels, entailmentPos, contraPos, isMultiLabel);

            List<TopClassEntry> topClassEntries = Arrays.stream(result.normalizedScores)
                    .boxed()
                    .sorted(Comparator.reverseOrder())
                    .limit(labels.length)
                    .map(score -> new ResultEntry(getLabelFromScore(score, labels, result.normalizedScores), score))
                    .map(entry -> new TopClassEntry(entry.label(), entry.score()))
                    .collect(Collectors.toList());

            return new NlpClassificationInferenceResults(
                    topClassEntries.get(0).getLabel(),
                    topClassEntries,
                    Optional.ofNullable(resultsField).orElse(DEFAULT_RESULTS_FIELD),
                    result.normalizedScores[0],
                    tokenization.anyTruncated()
            );
        }

        private static Result getResult(PyTorchInferenceResult pyTorchResult, String[] labels, int entailmentPos, int contraPos, boolean isMultiLabel) {
            final double[] normalizedScores;
            if (isMultiLabel) {
                normalizedScores = getMultiLabelScores(pyTorchResult, entailmentPos, contraPos, labels);
            } else {
                normalizedScores = getSingleLabelScores(pyTorchResult.getInferenceResult()[0], entailmentPos);
            }

            return new Result(new NlpClassificationInferenceResults(), normalizedScores);
        }

        private static double[] getMultiLabelScores(PyTorchInferenceResult pyTorchResult, int entailmentPos, int contraPos, String[] labels) {
            double[] normalizedScores = new double[pyTorchResult.getInferenceResult()[0].length];
            int v = 0;
            for (double[] vals : pyTorchResult.getInferenceResult()[0]) {
                if (vals.length != MIN_INNER_CLASSIFICATION_RESULT_LENGTH) {
                    throw new ElasticsearchStatusException(
                            "Expected exactly [{}] values in inner zero shot classification result; got [{}]",
                            RestStatus.INTERNAL_SERVER_ERROR,
                            MIN_INNER_CLASSIFICATION_RESULT_LENGTH,
                            vals.length
                    );
                }
                normalizedScores[v++] = NlpHelpers.convertToProbabilitiesBySoftMax(
                        new double[] { vals[entailmentPos], vals[contraPos] }
                )[0];
            }
            return normalizedScores;
        }

        private static double[] getSingleLabelScores(double[] entailmentScores, int entailmentPos) {
            double[] normalizedScores = new double[entailmentScores.length];
            int v = 0;
            for (double val : entailmentScores) {
                if (entailmentScores.length != MIN_INNER_CLASSIFICATION_RESULT_LENGTH) {
                    throw new ElasticsearchStatusException(
                            "Expected exactly [{}] values in inner zero shot classification result; got [{}]",
                            RestStatus.INTERNAL_SERVER_ERROR,
                            MIN_INNER_CLASSIFICATION_RESULT_LENGTH,
                            entailmentScores.length
                    );
                }
                normalizedScores[v++] = entailmentScores[entailmentPos];
            }
            return NlpHelpers.convertToProbabilitiesBySoftMax(normalizedScores);
        }

        private static String getLabelFromScore(double score, String[] labels, double[] scores) {
            return labels[IntStream.range(0, scores.length)
                    .boxed()
                    .filter(i -> scores[i] == score)
                    .findFirst()
                    .getAsInt()];
        }
    }
}