package org.elasticsearch.index.query;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

import org.apache.lucene.index.Term;
import org.apache.lucene.index.memory.MemoryIndex;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.test.AbstractQueryTestCase;
import org.elasticsearch.xcontent.XContentParseException;
import org.elasticsearch.xcontent.XContentType;
import org.hamcrest.Matchers;

import static org.elasticsearch.search.SearchModule.INDICES_MAX_NESTED_DEPTH_SETTING;

public class WrapperQueryBuilderTests extends AbstractQueryTestCase<WrapperQueryBuilder> {

    public static final String NAME = "wrapper";

    @Override
    protected boolean supportsBoost() {
        return false;
    }

    @Override
    protected boolean supportsQueryName() {
        return false;
    }

    @Override
    protected boolean builderGeneratesCacheableQueries() {
        return false;
    }

    @Override
    protected WrapperQueryBuilder doCreateTestQueryBuilder() {
        QueryBuilder wrappedQuery = RandomQueryBuilder.createQuery(random());
        BytesReference bytes = initBytes(wrappedQuery);
        return new WrapperQueryBuilder(bytes);
    }
    
    private BytesReference initBytes(QueryBuilder wrappedQuery) {
        try {
            return XContentHelper.toXContent(wrappedQuery, XContentType.JSON, false);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    protected void doAssertLuceneQuery(WrapperQueryBuilder queryBuilder, Query query, SearchExecutionContext context) throws IOException {
        QueryBuilder innerQuery = queryBuilder.rewrite(createSearchExecutionContext());
        Query expected = rewrite(innerQuery.toQuery(context));
        assertEquals(expected, query);
    }

    public void testIllegalArgument() {
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((byte[]) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(new byte[0]));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((String) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(""));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((BytesReference) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(new BytesArray(new byte[0])));
    }

    /**
     * Replace the generic test from superclass, wrapper query only expects
     * to find `query` field with nested query and should throw exception for
     * anything else.
     */
    @Override
    public void testUnknownField() {
        String json = "{ \"" + NAME + "\" : {\"bogusField\" : \"someValue\"} }";
        ParsingException e = expectThrows(ParsingException.class, () -> parseQuery(json));
        assertTrue(e.getMessage().contains("bogusField"));
    }

    public void testFromJson() throws IOException {
        String json = """
            {
              "%s" : {
                "query" : "e30="
              }
            }""".formatted(NAME);

        WrapperQueryBuilder parsed = (WrapperQueryBuilder) parseQuery(json);
        checkGeneratedJson(json, parsed);
        assertEquals(new String(parsed.source(), StandardCharsets.UTF_8), "{}");
    }

    @Override
    public void testMustRewrite() throws IOException {
        TermQueryBuilder tqb = new TermQueryBuilder(TEXT_FIELD_NAME, "bar");
        WrapperQueryBuilder qb = new WrapperQueryBuilder(tqb.toString());
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> qb.toQuery(createSearchExecutionContext())
        );
        assertEquals("this query must be rewritten first", e.getMessage());
        QueryBuilder rewrite = qb.rewrite(createSearchExecutionContext());
        assertEquals(tqb, rewrite);
    }

    public void testRewriteWithInnerName() throws IOException {
        QueryBuilder builder = new WrapperQueryBuilder("""
            { "match_all" : {"_name" : "foobar"}}""");
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();
        assertEquals(new MatchAllQueryBuilder().queryName("foobar"), builder.rewrite(searchExecutionContext));
        builder = new WrapperQueryBuilder("""
            { "match_all" : {"_name" : "foobar"}}""").queryName("outer");
        assertEquals(
            new BoolQueryBuilder().must(new MatchAllQueryBuilder().queryName("foobar")).queryName("outer"),
            builder.rewrite(searchExecutionContext)
        );
    }

    public void testRewriteWithInnerBoost() throws IOException {
        final TermQueryBuilder query = new TermQueryBuilder(TEXT_FIELD_NAME, "bar").boost(2);
        QueryBuilder builder = new WrapperQueryBuilder(query.toString());
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();
        assertEquals(query, builder.rewrite(searchExecutionContext));
        builder = new WrapperQueryBuilder(query.toString()).boost(3);
        assertEquals(new BoolQueryBuilder().must(query).boost(3), builder.rewrite(searchExecutionContext));
    }

    public void testRewriteInnerQueryToo() throws IOException {
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();

        QueryBuilder qb = new WrapperQueryBuilder(
            new WrapperQueryBuilder(new TermQueryBuilder(TEXT_FIELD_NAME, "bar").toString()).toString()
        );
        assertEquals(new TermQuery(new Term(TEXT_FIELD_NAME, "bar")), qb.rewrite(searchExecutionContext).toQuery(searchExecutionContext));
        qb = new WrapperQueryBuilder(
            new WrapperQueryBuilder(new WrapperQueryBuilder(new TermQueryBuilder(TEXT_FIELD_NAME, "bar").toString()).toString()).toString()
        );
        assertEquals(new TermQuery(new Term(TEXT_FIELD_NAME, "bar")), qb.rewrite(searchExecutionContext).toQuery(searchExecutionContext));

        qb = new WrapperQueryBuilder(new BoolQueryBuilder().toString());
        assertEquals(new MatchAllDocsQuery(), qb.rewrite(searchExecutionContext).toQuery(searchExecutionContext));
    }

    @Override
    protected Query rewrite(Query query) throws IOException {
        if (query != null) {
            MemoryIndex idx = new MemoryIndex();
            return idx.createSearcher().rewrite(query);
        }
        return new MatchAllDocsQuery(); // null == *:*
    }

    @Override
    protected WrapperQueryBuilder createQueryWithInnerQuery(QueryBuilder queryBuilder) {
        return new WrapperQueryBuilder(Strings.toString(queryBuilder));
    }

    public void testMaxNestedDepth() throws IOException {
        BoolQueryBuilderTest boolQueryBuilderTest = new BoolQueryBuilderTest();
        BoolQueryBuilder boolQuery = boolQueryBuilderTest.createQueryWithInnerQuery(new MatchAllQueryBuilder());
        int maxDepth = randomIntBetween(3, 5);
        AbstractQueryBuilder.setMaxNestedDepth(maxDepth);
        for (int i = 1; i < maxDepth - 1; i++) {
            boolQuery = boolQueryBuilderTest.createQueryWithInnerQuery(boolQuery);
        }
        WrapperQueryBuilder query = new WrapperQueryBuilder(Strings.toString(boolQuery));
        AbstractQueryBuilder.setMaxNestedDepth(maxDepth);
        try {
            query.rewrite(createSearchExecutionContext());
            WrapperQueryBuilder q = new WrapperQueryBuilder(Strings.toString(boolQueryBuilderTest.createQueryWithInnerQuery(boolQuery)));
            IllegalArgumentException e = expectThrows(XContentParseException.class, () -> q.rewrite(createSearchExecutionContext()));
            while (e.getCause() != null) {
                assertThat(e.getCause(), Matchers.instanceOf(IllegalArgumentException.class));
                e = (IllegalArgumentException) e.getCause();
            }

            assertEquals(
                "The nested depth of the query exceeds the maximum nested depth for queries set in [%s]".formatted(INDICES_MAX_NESTED_DEPTH_SETTING.getKey()),
                e.getMessage()
            );
        } finally {
            AbstractQueryBuilder.setMaxNestedDepth(INDICES_MAX_NESTED_DEPTH_SETTING.getDefault(Settings.EMPTY));
        }
    }
}