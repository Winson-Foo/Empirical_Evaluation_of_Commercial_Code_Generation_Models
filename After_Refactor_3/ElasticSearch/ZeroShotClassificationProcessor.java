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
    private final List<String> labels;
    private final String hypothesisTemplate;
    private final boolean isMultiLabel;
    private final String resultsField;

    public ZeroShotClassificationProcessor(NlpTokenizer tokenizer, ZeroShotClassificationConfig config) {
        super(tokenizer);
        List<String> lowerCased = config.getClassificationLabels().stream().map(String::toLowerCase).toList();
        this.entailmentPos = lowerCased.indexOf("entailment");
        this.contraPos = lowerCased.indexOf("contradiction");
        if (entailmentPos == -1 || contraPos == -1) {
            throw ExceptionsHelper.badRequestException(
                "zero_shot_classification requires [entailment] and [contradiction] in classification_labels"
            );
        }
        this.labels = List.copyOf(config.getLabels().orElse(List.of()));
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
        final List<String> labelsValue;
        if (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig) {
            labelsValue = List.copyOf(zeroShotConfig.getLabels().orElse(List.of()));
        } else {
            labelsValue = this.labels;
        }
        if (labelsValue == null || labelsValue.isEmpty()) {
            throw ExceptionsHelper.badRequestException("zero_shot_classification requires non-empty [labels]");
        }
        return new RequestBuilder(tokenizer, labelsValue, hypothesisTemplate);
    }

    @Override
    public NlpTask.ResultProcessor getResultProcessor(NlpConfig nlpConfig) {
        final List<String> labelsValue;
        final boolean isMultiLabelValue;
        final String resultsFieldValue;
        if (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig) {
            labelsValue = List.copyOf(zeroShotConfig.getLabels().orElse(List.of()));
            isMultiLabelValue = zeroShotConfig.isMultiLabel();
            resultsFieldValue = zeroShotConfig.getResultsField();
        } else {
            labelsValue = this.labels;
            isMultiLabelValue = this.isMultiLabel;
            resultsFieldValue = this.resultsField;
        }
        return new ResultProcessor(entailmentPos, contraPos, labelsValue, isMultiLabelValue, resultsFieldValue);
    }

    record RequestBuilder(
        NlpTokenizer tokenizer,
        List<String> labels,
        String hypothesisTemplate
    ) implements NlpTask.RequestBuilder {

        @Override
        public NlpTask.Request buildRequest(
            List<String> inputs,
            String requestId,
            Tokenization.Truncate truncate,
            int span
        ) throws IOException {
            if (inputs.size() > 1) {
                throw ExceptionsHelper.badRequestException("Unable to do zero-shot classification on more than one text input at a time");
            }
            if (span > -1) {
                throw ExceptionsHelper.badRequestException("Unable to span zero-shot classification on long text input");
            }
            List<TokenizationResult.Tokens> tokenizations = new ArrayList<>(labels.size());
            int seqId = 0;
            NlpTokenizer.InnerTokenization firstSequenceTokenization = tokenizer.innerTokenize(inputs.get(0));
            for (String label : labels) {
                tokenizations.add(
                    tokenizer.tokenize(
                        inputs.get(0),
                        firstSequenceTokenization,
                        LoggerMessageFormat.format(null, hypothesisTemplate, label),
                        truncate,
                        seqId++
                    )
                );
            }
            TokenizationResult result = tokenizer.buildTokenizationResult(tokenizations);
            return result.buildRequest(requestId, truncate);
        }
    }

    static class NormalizedScores {
        public final double[] scores;

        public NormalizedScores(double[] scores) {
            this.scores = scores;
        }
    }

    static class PyTorchResultParser {
        public static NormalizedScores parse(PyTorchInferenceResult pyTorchResult, int entailmentPos, int contraPos) {
            if (pyTorchResult.getInferenceResult().length < 1) {
                throw new ElasticsearchStatusException("Zero shot classification result has no data", RestStatus.INTERNAL_SERVER_ERROR);
            }
            double[] normalizedScores;
            if (pyTorchResult.getInferenceResult()[0].length != 3) {
                throw new ElasticsearchStatusException(
                    "Expected exactly [3] values in zero shot classification result; got [{}]",
                    RestStatus.INTERNAL_SERVER_ERROR,
                    pyTorchResult.getInferenceResult().length
                );
            }
            final double[] entailmentScores = new double[pyTorchResult.getInferenceResult().length];
            int v = 0;
            for (double[] vals : pyTorchResult.getInferenceResult()) {
                entailmentScores[v++] = vals[entailmentPos];
            }
            normalizedScores = NlpHelpers.convertToProbabilitiesBySoftMax(entailmentScores);
            return new NormalizedScores(normalizedScores);
        }
    }

    static class InferenceResultsCreator {
        public static InferenceResults create(
            TokenizationResult tokenization,
            PyTorchInferenceResult pyTorchResult,
            List<String> labels,
            boolean isMultiLabel,
            String resultsField,
            int entailmentPos,
            int contraPos
        ) {
            NormalizedScores normalizedScores = PyTorchResultParser.parse(pyTorchResult, entailmentPos, contraPos);
            int[] sortedIndices = IntStream.range(0, normalizedScores.scores.length)
                .boxed()
                .sorted(Comparator.comparing(i -> normalizedScores.scores[(int) i]).reversed())
                .mapToInt(i -> i)
                .toArray();
            String topLabel = labels.get(sortedIndices[0]);
            List<TopClassEntry> topClasses = Arrays.stream(sortedIndices)
                .mapToObj(i -> new TopClassEntry(labels.get(i), normalizedScores.scores[i]))
                .collect(Collectors.toList());
            return new NlpClassificationInferenceResults(
                topLabel,
                topClasses,
                Optional.ofNullable(resultsField).orElse(DEFAULT_RESULTS_FIELD),
                normalizedScores.scores[sortedIndices[0]],
                tokenization.anyTruncated()
            );
        }
    }

    record ResultProcessor(
        int entailmentPos,
        int contraPos,
        List<String> labels,
        boolean isMultiLabel,
        String resultsField
    ) implements NlpTask.ResultProcessor {

        @Override
        public InferenceResults processResult(TokenizationResult tokenization, PyTorchInferenceResult pyTorchResult) {
            if (isMultiLabel) {
                throw new UnsupportedOperationException("Multiple labels not supported yet");
            }
            return InferenceResultsCreator.create(
                tokenization,
                pyTorchResult,
                labels,
                isMultiLabel,
                resultsField,
                entailmentPos,
                contraPos
            );
        }
    }
}