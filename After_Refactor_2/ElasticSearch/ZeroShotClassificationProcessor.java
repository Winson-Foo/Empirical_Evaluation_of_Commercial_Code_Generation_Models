// Import statements

public class ZeroShotClassificationProcessor extends NlpTask.Processor {
    
    private final int entailmentPos;
    private final int contraPos;
    private final String[] labels;
    private final String hypothesisTemplate;
    private final boolean isMultiLabel;
    private final String resultsField;
    
    public ZeroShotClassificationProcessor(NlpTokenizer tokenizer, ZeroShotClassificationConfig config) {
        super(tokenizer);
        List<String> lowerCased = config.getClassificationLabels().stream().map(String::toLowerCase).collect(Collectors.toList());
        this.entailmentPos = lowerCased.indexOf("entailment");
        this.contraPos = lowerCased.indexOf("contradiction");
        if (entailmentPos == -1 || contraPos == -1) {
            throw ExceptionsHelper.badRequestException(
                "zero_shot_classification requires [entailment] and [contradiction] in classification_labels"
            );
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
        if (labelsValue == null || labelsValue.length == 0) {
            throw ExceptionsHelper.badRequestException("zero_shot_classification requires non-empty [labels]");
        }
        return new RequestBuilder(tokenizer, labelsValue, hypothesisTemplate);
    }

    @Override
    public NlpTask.ResultProcessor getResultProcessor(NlpConfig nlpConfig) {
        final String[] labelsValue = getLabelsValue(nlpConfig);
        final boolean isMultiLabelValue = getIsMultiLabelValue(nlpConfig);
        final String resultsFieldValue = getResultsFieldValue(nlpConfig);
        return new ResultProcessor(entailmentPos, contraPos, labelsValue, isMultiLabelValue, resultsFieldValue);
    }

    private String[] getLabelsValue(NlpConfig nlpConfig) {
        Optional<List<String>> optionalLabels = Optional.empty();
        if (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig) {
            optionalLabels = zeroShotConfig.getLabels();
        }
        return optionalLabels.orElse(List.of()).toArray(new String[0]);
    }

    private boolean getIsMultiLabelValue(NlpConfig nlpConfig) {
        if (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig) {
            return zeroShotConfig.isMultiLabel();
        } 
        return isMultiLabel;
    }

    private String getResultsFieldValue(NlpConfig nlpConfig) {
        if (nlpConfig instanceof ZeroShotClassificationConfig zeroShotConfig) {
            return zeroShotConfig.getResultsField();
        } 
        return resultsField;
    }

    record RequestBuilder(NlpTokenizer tokenizer, String[] labels, String hypothesisTemplate) implements NlpTask.RequestBuilder {

        @Override
        public NlpTask.Request buildRequest(List<String> inputs, String requestId, Tokenization.Truncate truncate, int span)
            throws IOException {
            if (inputs.size() > 1) {
                throw ExceptionsHelper.badRequestException("Unable to do zero-shot classification on more than one text input at a time");
            }
            if (span > -1) {
                throw ExceptionsHelper.badRequestException("Unable to span zero-shot classification on long text input");
            }
            List<TokenizationResult.Tokens> tokenizations = new ArrayList<>(labels.length);
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

    record ResultProcessor(int entailmentPos, int contraPos, String[] labels, boolean isMultiLabel, String resultsField)
        implements NlpTask.ResultProcessor {

        @Override
        public InferenceResults processResult(TokenizationResult tokenization, PyTorchInferenceResult pyTorchResult) {
            validatePyTorchResult(pyTorchResult);
            final double[] normalizedScores = getNormalizedScores(pyTorchResult);
            int[] sortedIndices = sortIndices(normalizedScores);
            return getInferenceResults(sortedIndices, normalizedScores, tokenization.anyTruncated());
        }

        private void validatePyTorchResult(PyTorchInferenceResult pyTorchResult) {
            if (pyTorchResult.getInferenceResult().length < 1) {
                throw new ElasticsearchStatusException("Zero shot classification result has no data", RestStatus.INTERNAL_SERVER_ERROR);
            }
            // TODO only the first entry in the batch result is verified and checked. Implement for all in batch
            if (pyTorchResult.getInferenceResult()[0].length != labels.length) {
                throw new ElasticsearchStatusException(
                    "Expected exactly [{}] values in zero shot classification result; got [{}]",
                    RestStatus.INTERNAL_SERVER_ERROR,
                    labels.length,
                    pyTorchResult.getInferenceResult().length
                );
            }
        }

        private double[] getNormalizedScores(PyTorchInferenceResult pyTorchResult) {
            if (isMultiLabel) {
                return getMultiLabelScores(pyTorchResult);
            } 
            return getSingleLabelScores(pyTorchResult);
        }

        private double[] getSingleLabelScores(PyTorchInferenceResult pyTorchResult) {
            double[] entailmentScores = new double[pyTorchResult.getInferenceResult()[0].length];
            int v = 0;
            for (double[] vals : pyTorchResult.getInferenceResult()[0]) {
                if (vals.length != 3) {
                    throw new ElasticsearchStatusException(
                        "Expected exactly [{}] values in inner zero shot classification result; got [{}]",
                        RestStatus.INTERNAL_SERVER_ERROR,
                        3,
                        vals.length
                    );
                }
                entailmentScores[v++] = vals[entailmentPos];
            }
            return NlpHelpers.convertToProbabilitiesBySoftMax(entailmentScores);
        }

        private double[] getMultiLabelScores(PyTorchInferenceResult pyTorchResult) {
            double[] normalizedScores = new double[pyTorchResult.getInferenceResult()[0].length];
            int v = 0;
            for (double[] vals : pyTorchResult.getInferenceResult()[0]) {
                if (vals.length != 3) {
                    throw new ElasticsearchStatusException(
                        "Expected exactly [{}] values in inner zero shot classification result; got [{}]",
                        RestStatus.INTERNAL_SERVER_ERROR,
                        3,
                        vals.length
                    );
                }
                // assume entailment is `0`, softmax between entailment and contradiction
                normalizedScores[v++] = NlpHelpers.convertToProbabilitiesBySoftMax(
                    new double[] { vals[entailmentPos], vals[contraPos] }
                )[0];
            }
            return normalizedScores;
        }

        private int[] sortIndices(double[] normalizedScores) {
            return IntStream.range(0, normalizedScores.length)
                .boxed()
                .sorted(Comparator.comparing(i -> normalizedScores[(Integer) i]).reversed())
                .mapToInt(i -> i)
                .toArray();
        }

        private InferenceResults getInferenceResults(int[] sortedIndices, double[] normalizedScores, boolean anyTruncated) {
            String topClassLabel = labels[sortedIndices[0]];
            List<TopClassEntry> topClassEntries = Arrays.stream(sortedIndices)
                .mapToObj(i -> new TopClassEntry(labels[i], normalizedScores[i]))
                .collect(Collectors.toList());
            String resultsField = Optional.ofNullable(resultsField).orElse(DEFAULT_RESULTS_FIELD);
            double topClassScore = normalizedScores[sortedIndices[0]];
            return new NlpClassificationInferenceResults(
                topClassLabel,
                topClassEntries,
                resultsField,
                topClassScore,
                anyTruncated
            );
        }
    }
}