package org.elasticsearch.rest.action.search;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.elasticsearch.action.search.SearchScrollRequest;
import org.elasticsearch.client.internal.node.NodeClient;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.RestChannelConsumer;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.Scope;
import org.elasticsearch.rest.ServerlessScope;
import org.elasticsearch.rest.action.RestChunkedToXContentListener;
import org.elasticsearch.search.Scroll;
import org.elasticsearch.xcontent.XContentParseException;

import static org.elasticsearch.core.TimeValue.parseTimeValue;
import static org.elasticsearch.rest.RestRequest.Method.GET;
import static org.elasticsearch.rest.RestRequest.Method.POST;

@ServerlessScope(Scope.PUBLIC)
public class RestSearchScrollAction extends BaseRestHandler {

    private static final String SCROLL_ID_PARAM = "scroll_id";
    private static final String SCROLL_PARAM = "scroll";
    private static final String SEARCH_SCROLL_ACTION_NAME = "search_scroll_action";
    private static final Set<String> RESPONSE_PARAMS = Collections.singleton(RestSearchAction.TOTAL_HITS_AS_INT_PARAM);

    @Override
    public String getName() {
        return SEARCH_SCROLL_ACTION_NAME;
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
        String scrollId = request.param(SCROLL_ID_PARAM);
        SearchScrollRequest searchScrollRequest = new SearchScrollRequest();
        searchScrollRequest.scrollId(scrollId);
        return searchScrollRequest;
    }

    private void parseRequestBody(final RestRequest request, final SearchScrollRequest searchScrollRequest) {
        request.withContentOrSourceParamParserOrNull(xContentParser -> {
            if (xContentParser != null) {
                try {
                    searchScrollRequest.fromXContent(xContentParser);
                } catch (IOException | XContentParseException e) {
                    throw new IllegalArgumentException("Failed to parse request body", e);
                }
            }
        });
    }

    @Override
    protected Set<String> responseParams() {
        return RESPONSE_PARAMS;
    }
}