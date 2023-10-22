package org.elasticsearch.rest.action.search;

import org.elasticsearch.action.search.SearchScrollRequest;
import org.elasticsearch.client.internal.node.NodeClient;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.Scope;
import org.elasticsearch.rest.ServerlessScope;
import org.elasticsearch.rest.action.RestChunkedToXContentListener;
import org.elasticsearch.search.Scroll;
import org.elasticsearch.xcontent.XContentParseException;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.elasticsearch.core.TimeValue.parseTimeValue;
import static org.elasticsearch.rest.RestRequest.Method.GET;
import static org.elasticsearch.rest.RestRequest.Method.POST;

@ServerlessScope(Scope.PUBLIC)
public class RestSearchScrollHandler extends BaseRestHandler {
    private static final Set<String> RESPONSE_PARAMS = Collections.singleton(RestSearchAction.TOTAL_HITS_AS_INT_PARAM);
    private static final String SCROLL_ENDPOINT = "/_search/scroll";
    private static final String SCROLL_ID_PARAM = "scroll_id";
    private static final String SCROLL_PARAM = "scroll";

    @Override
    public String getName() {
        return "search_scroll_handler";
    }

    @Override
    public List<Route> routes() {
        return List.of(
                new Route(GET, SCROLL_ENDPOINT),
                new Route(POST, SCROLL_ENDPOINT),
                new Route(GET, SCROLL_ENDPOINT + "/{" + SCROLL_ID_PARAM + "}"),
                new Route(POST, SCROLL_ENDPOINT + "/{" + SCROLL_ID_PARAM + "}")
        );
    }

    @Override
    public RestChannelConsumer prepareRequest(final RestRequest request, final NodeClient client) throws IOException {
        SearchScrollRequest searchScrollRequest = createSearchScrollRequest(request);
        parseRequestBody(request, searchScrollRequest);
        return channel -> client.searchScroll(searchScrollRequest, new RestChunkedToXContentListener<>(channel));
    }

    @Override
    protected Set<String> responseParams() {
        return RESPONSE_PARAMS;
    }

    private SearchScrollRequest createSearchScrollRequest(RestRequest request) {
        String scrollId = request.param(SCROLL_ID_PARAM);
        SearchScrollRequest searchScrollRequest = new SearchScrollRequest();
        searchScrollRequest.scrollId(scrollId);
        String scroll = request.param(SCROLL_PARAM);
        if (scroll != null) {
            searchScrollRequest.scroll(new Scroll(parseTimeValue(scroll, null, "scroll")));
        }
        return searchScrollRequest;
    }

    private void parseRequestBody(RestRequest request, SearchScrollRequest searchScrollRequest) throws IOException {
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
}