package org.elasticsearch.index.query;

import java.util.Collections;
import java.util.Map;

import org.apache.lucene.search.Query;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.script.MockScriptEngine;
import org.elasticsearch.script.Script;
import org.elasticsearch.script.ScriptType;
import org.elasticsearch.test.AbstractQueryTestCase;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for the {@link ScriptQueryBuilder} class.
 */
public class ScriptQueryBuilderTests extends AbstractQueryTestCase<ScriptQueryBuilder> {
    private static final String MOCK_LANG = "mockscript";
    private static final String MOCK_SCRIPT_BODY = "1";

    @Override
    protected ScriptQueryBuilder doCreateTestQueryBuilder() {
        Map<String, Object> params = Collections.emptyMap();
        return new ScriptQueryBuilder(
                new Script(ScriptType.INLINE, MockScriptEngine.NAME, MOCK_SCRIPT_BODY, params));
    }

    /**
     * This query is generally not cacheable.
     */
    @Override
    public void testCacheability() throws Exception {
        SearchExecutionContext context = createSearchExecutionContext();
        ScriptQueryBuilder queryBuilder = createTestQueryBuilder();
        QueryBuilder rewriteQuery = rewriteQuery(queryBuilder, new SearchExecutionContext(context));
        assertNotNull(rewriteQuery.toQuery(context));
        assertFalse("query should not be cacheable", context.isCacheable());
    }

    public void testIllegalConstructorArg() {
        assertThatThrownBy(() -> new ScriptQueryBuilder((Script) null))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessage("script cannot be null");
    }

    public void testFromJsonVerbose() throws Exception {
        String json = getXContentBuilderJsonString();
        ScriptQueryBuilder parsed = (ScriptQueryBuilder) parseQuery(json);
        checkGeneratedJson(json, parsed);
        assertThat(parsed.script().getLang()).isEqualTo(MOCK_LANG);
    }

    public void testFromJson() throws Exception {
        String json = getXContentBuilderJsonString();
        ScriptQueryBuilder parsed = (ScriptQueryBuilder) parseQuery(json);
        assertThat(parsed.script().getIdOrCode()).isEqualTo(MOCK_SCRIPT_BODY);
    }

    public void testArrayOfScriptsException() {
        String json = """
            {
              "script" : {
                "script" : [ 
                {
                  "source" : "5",
                  "lang" : "mockscript"
                },
                {
                  "source" : "6",
                  "lang" : "mockscript"
                }
             ]  }
            }""";

        assertThatThrownBy(() -> parseQuery(json))
                .isInstanceOf(ParsingException.class)
                .hasMessageContaining("does not support an array of scripts");
    }

    public void testDisallowExpensiveQueries() {
        SearchExecutionContext context = mock(SearchExecutionContext.class);
        when(context.allowExpensiveQueries()).thenReturn(false);

        ScriptQueryBuilder queryBuilder = doCreateTestQueryBuilder();
        assertThatThrownBy(() -> queryBuilder.toQuery(context))
                .isInstanceOf(ElasticsearchException.class)
                .hasMessage("[script] queries cannot be executed when 'search.allow_expensive_queries' is set to false.");
    }

    /**
     * Returns the JSON representation of a script query.
     *
     * @return the JSON string
     */
    private String getXContentBuilderJsonString() throws Exception {
        XContentBuilder builder = createXContentBuilder().startObject()
                .startObject("script")
                .field("script", MOCK_SCRIPT_BODY)
                .field("boost", 1.0f)
                .field("_name", "PcKdEyPOmR")
                .endObject()
                .endObject();
        return builder.prettyPrint().string();
    }
}