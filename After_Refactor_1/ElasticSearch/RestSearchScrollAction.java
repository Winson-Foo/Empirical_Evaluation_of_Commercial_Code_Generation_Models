public class RestSearchScrollAction extends BaseRestHandler {
    private static final Set<String> RESPONSE_PARAMS = Collections.singleton(RestSearchAction.TOTAL_HITS_AS_INT_PARAM);

    @Override
    public String getName() {
        return "search_scroll_action";
    }

    @Override
    public List<Route> routes() {
        return List.of(
            new Route(GET, "/_search/scroll"),
            new Route(POST, "/_search/scroll"),
            new Route(GET, "/_search/scroll/{scroll_id}"),
            new Route(POST, "/_search/scroll/{scroll_id}")
        );
    }

    @Override
    public RestChannelConsumer prepareRequest(final RestRequest request, final NodeClient client) throws IOException {

        SearchScrollRequest searchScrollRequest = createSearchScrollRequest(request);

        parseRequestBody(request, searchScrollRequest);

        return channel -> client.searchScroll(searchScrollRequest, new RestChunkedToXContentListener<>(channel));
    }

    private SearchScrollRequest createSearchScrollRequest(final RestRequest request) {
        String scrollId = request.param("scroll_id");
        return new SearchScrollRequest().scrollId(scrollId);
    }

    private void parseRequestBody(final RestRequest request, final SearchScrollRequest searchScrollRequest) throws IOException {
        request.withContentOrSourceParamParserOrNull(xContentParser -> {
            if (xContentParser != null) {
                try {
                    searchScrollRequest.fromXContent(xContentParser);
                } catch (IOException | XContentParseException e) {
                    throw new IllegalArgumentException("Failed to parse request body", e);
                }
            }
        });

        String scroll = request.param("scroll");
        if (scroll != null) {
            searchScrollRequest.scroll(new Scroll(parseScrollParameter(scroll)));
        }
    }

    private TimeValue parseScrollParameter(final String scroll) {
        return parseTimeValue(scroll, null, "scroll");
    }

    @Override
    protected Set<String> responseParams() {
        return RESPONSE_PARAMS;
    }
}