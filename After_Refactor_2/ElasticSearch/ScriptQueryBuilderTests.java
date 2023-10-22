package org.elasticsearch.index.query;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import org.apache.lucene.search.Query;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.script.MockScriptEngine;
import org.elasticsearch.script.Script;
import org.elasticsearch.script.ScriptType;
import org.elasticsearch.test.AbstractQueryTestCase;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

public class ScriptQueryBuilderTests extends AbstractQueryTestCase<ScriptQueryBuilder> {

    private static final String LANG = "mockscript";
    private static final String SCRIPT = "5";

    @Override
    protected ScriptQueryBuilder createTestQueryBuilder() {
        Map<String, Object> params = Collections.emptyMap();
        Script script = new Script(ScriptType.INLINE, LANG, SCRIPT, params);
        return new ScriptQueryBuilder(script);
    }

    @Override
    protected boolean builderGeneratesCacheableQueries() {
        return false;
    }

    @Override
    protected void doAssertLuceneQuery(ScriptQueryBuilder queryBuilder, Query query, SearchExecutionContext context) throws IOException {
        assertThat(query, instanceOf(ScriptQueryBuilder.ScriptQuery.class));
    }

    @ParameterizedTest
    @EnumSource(ScriptType.class)
    void testFromJson(ScriptType scriptType) throws IOException {
        String json = "{" +
                "\"script\": {" +
                "\"script\": \"" + SCRIPT + "\"" +
                "}" +
                "}";

        Script script = new Script(scriptType, LANG, SCRIPT, Collections.emptyMap());
        ScriptQueryBuilder expected = new ScriptQueryBuilder(script);

        ScriptQueryBuilder actual = (ScriptQueryBuilder) parseQuery(json);
        assertEquals(expected, actual);
    }

    @Test
    void testFromJsonVerbose() throws IOException {
        String json = "{" +
                "\"script\": {" +
                "\"script\": {" +
                "\"source\": \"" + SCRIPT + "\"," +
                "\"lang\": \"" + LANG + "\"" +
                "}," +
                "\"boost\": 1.0," +
                "\"_name\": \"PcKdEyPOmR\"" +
                "}" +
                "}";

        Script script = new Script(ScriptType.INLINE, LANG, SCRIPT, Collections.emptyMap());
        ScriptQueryBuilder expected = new ScriptQueryBuilder(script).boost(1.0f).queryName("PcKdEyPOmR");

        ScriptQueryBuilder actual = (ScriptQueryBuilder) parseQuery(json);
        assertEquals(expected, actual);

        assertEquals(LANG, actual.script().getLang());
    }

    @Nested
    class ScriptArrayTests {

        @Test
        void shouldThrowParsingExceptionForArrayOfScripts() {
            String json = "{" +
                    "\"script\": {" +
                    "\"script\": [" +
                    "{" +
                    "\"source\": \"" + SCRIPT + "\"," +
                    "\"lang\": \"" + LANG + "\"" +
                    "}," +
                    "{" +
                    "\"source\": \"6\"," +
                    "\"lang\": \"mockscript\"" +
                    "}" +
                    "]" +
                    "}" +
                    "}";

            ParsingException e = expectThrows(ParsingException.class, () -> parseQuery(json));
            assertThat(e.getMessage(), containsString("does not support an array of scripts"));
        }

    }

    @Override
    protected Map<String, String> getObjectsHoldingArbitraryContent() {
        return Collections.singletonMap(Script.PARAMS_PARSE_FIELD.getPreferredName(), null);
    }

    @Override
    public void testCacheability() throws IOException {
        ScriptQueryBuilder queryBuilder = createTestQueryBuilder();
        SearchExecutionContext context = createSearchExecutionContext();
        QueryBuilder rewriteQuery = rewriteQuery(queryBuilder, new SearchExecutionContext(context));
        assertNotNull(rewriteQuery.toQuery(context));
        assertFalse("query should not be cacheable: " + queryBuilder.toString(), context.isCacheable());
    }

    @Test
    void testDisallowExpensiveQueries() {
        SearchExecutionContext searchExecutionContext = mock(SearchExecutionContext.class);
        when(searchExecutionContext.allowExpensiveQueries()).thenReturn(false);

        ScriptQueryBuilder queryBuilder = createTestQueryBuilder();
        ElasticsearchException e = expectThrows(ElasticsearchException.class, () -> queryBuilder.toQuery(searchExecutionContext));
        assertEquals("[script] queries cannot be executed when 'search.allow_expensive_queries' is set to false.", e.getMessage());
    }
}