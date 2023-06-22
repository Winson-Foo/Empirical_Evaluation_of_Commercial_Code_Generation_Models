package org.elasticsearch.index.query;

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

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class ScriptQueryBuilderTests extends AbstractQueryTestCase<ScriptQueryBuilder> {

    private static final String SCRIPT_ID = "5";
    private static final String SCRIPT_LANG = "mockscript";
    private static final double BOOST = 1.0;
    private static final String SCRIPT_NAME = "PcKdEyPOmR";
    private static final String SCRIPT_SOURCE = "1";

    @Override
    protected ScriptQueryBuilder doCreateTestQueryBuilder() {
        Map<String, Object> params = Collections.emptyMap();
        Script script = new Script(ScriptType.INLINE, MockScriptEngine.NAME, SCRIPT_SOURCE, params);

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

    public void testIllegalConstructorArg() {
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> new ScriptQueryBuilder((Script) null));
        assertEquals("script cannot be null", e.getMessage());
    }

    public void testFromJsonVerbose() throws IOException {
        String json = """
            {
              "script" : {
                "script" : {
                  "source" : "%s",
                  "lang" : "%s"
                },
                "boost" : %f,
                "_name" : "%s"
              }
            }""".formatted(SCRIPT_SOURCE, SCRIPT_LANG, BOOST, SCRIPT_NAME);

        ScriptQueryBuilder parsed = (ScriptQueryBuilder) parseQuery(json);
        checkGeneratedJson(json, parsed);

        assertEquals("ScriptQueryBuilderSource", SCRIPT_SOURCE, parsed.script().getIdOrCode());
    }

    public void testFromJson() throws IOException {
        String json = """
            {
              "script" : {
                "script" : "%s",    
                "boost" : %f,
                "_name" : "%s"
              }
            }""".formatted(SCRIPT_SOURCE, BOOST, SCRIPT_NAME);

        ScriptQueryBuilder parsed = (ScriptQueryBuilder) parseQuery(json);

        assertEquals("ScriptQueryBuilderSource", SCRIPT_SOURCE, parsed.script().getIdOrCode());
    }

    public void testArrayOfScriptsException() {
        String json = """
            {
              "script" : {
                "script" : [ {
                  "source" : "%s",
                  "lang" : "%s"
                },
                {
                  "source" : "6",
                  "lang" : "%s"
                }
             ]  }
            }""".formatted(SCRIPT_SOURCE, SCRIPT_LANG, SCRIPT_LANG);

        ParsingException e = expectThrows(ParsingException.class, () -> parseQuery(json));
        assertThat(e.getMessage(), containsString("does not support an array of scripts"));
    }

    @Override
    protected Map<String, String> getObjectsHoldingArbitraryContent() {
        // script_score.script.params can contain arbitrary parameters. no error is expected when
        // adding additional objects within the params object.
        return Collections.singletonMap(Script.PARAMS_PARSE_FIELD.getPreferredName(), null);
    }

    /**
     * Check that this query is generally not cacheable
     */
    @Override
    public void testCacheability() throws IOException {
        ScriptQueryBuilder queryBuilder = createTestQueryBuilder();
        SearchExecutionContext context = createSearchExecutionContext();
        QueryBuilder rewriteQuery = rewriteQuery(queryBuilder, new SearchExecutionContext(context));
        assertNotNull(rewriteQuery.toQuery(context));
        assertFalse("query should not be cacheable: " + queryBuilder.toString(), context.isCacheable());
    }

    public void testDisallowExpensiveQueries() {
        SearchExecutionContext context = mock(SearchExecutionContext.class);
        when(context.allowExpensiveQueries()).thenReturn(false);

        ScriptQueryBuilder queryBuilder = doCreateTestQueryBuilder();
        ElasticsearchException e = expectThrows(ElasticsearchException.class, () -> queryBuilder.toQuery(context));
        assertEquals("[script] queries cannot be executed when 'search.allow_expensive_queries' is set to false.", e.getMessage());
    }
}