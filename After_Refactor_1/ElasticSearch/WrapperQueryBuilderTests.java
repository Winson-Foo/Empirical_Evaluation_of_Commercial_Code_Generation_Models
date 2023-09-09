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
import org.elasticsearch.common.bytes.BytesArray;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.test.AbstractQueryTestCase;
import org.elasticsearch.xcontent.XContentParseException;
import org.elasticsearch.xcontent.XContentType;
import org.hamcrest.Matchers;
import org.junit.Test;

import static org.elasticsearch.search.SearchModule.INDICES_MAX_NESTED_DEPTH_SETTING;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.expectThrows;

/**
 * Unit tests for {@link WrapperQueryBuilder}.
 */
public class WrapperQueryBuilderTests extends AbstractQueryTestCase<WrapperQueryBuilder> {

    /**
     * {@inheritDoc}
     */
    @Override
    protected boolean supportsBoost() {
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected boolean supportsQueryName() {
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected boolean builderGeneratesCacheableQueries() {
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected WrapperQueryBuilder doCreateTestQueryBuilder() {
        QueryBuilder wrappedQueryBuilder = RandomQueryBuilder.createQuery(random());
        BytesReference bytes;
        try {
            bytes = XContentHelper.toXContent(wrappedQueryBuilder, XContentType.JSON, false);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        return new WrapperQueryBuilder(bytes);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void doAssertLuceneQuery(WrapperQueryBuilder queryBuilder, Query query, SearchExecutionContext context) throws IOException {
        QueryBuilder innerQuery = queryBuilder.rewrite(createSearchExecutionContext());
        Query expected = rewrite(innerQuery.toQuery(context));
        assertThat(rewrite(query), Matchers.equalTo(expected));
    }

    /**
     * Tests that an exception is thrown for an illegal argument.
     */
    @Test
    public void testIllegalArgument() {
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((byte[]) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(new byte[0]));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((String) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(""));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder((BytesReference) null));
        expectThrows(IllegalArgumentException.class, () -> new WrapperQueryBuilder(new BytesArray(new byte[0])));
    }

    /**
     * Tests that an exception is thrown for an unknown field in the query.
     */
    @Test
    public void testUnknownField() {
        String json = "{ \"" + WrapperQueryBuilder.NAME + "\" : {\"bogusField\" : \"someValue\"} }";
        ParsingException e = expectThrows(ParsingException.class, () -> parseQuery(json));
        assertThat(e.getMessage(), Matchers.containsString("bogusField"));
    }

    /**
     * Tests JSON serialization and deserialization.
     */
    @Test
    public void testFromJson() throws IOException {
        String json = "{ \"wrapper\": {\"query\": \"e30=\"} }";
        WrapperQueryBuilder parsed = (WrapperQueryBuilder) parseQuery(json);
        checkGeneratedJson(json, parsed);
        assertEquals(json, "{}", new String(parsed.source(), StandardCharsets.UTF_8));
    }

    /**
     * Tests that the query is rewritten as expected.
     */
    @Test
    public void testMustRewrite() throws IOException {
        TermQueryBuilder termQueryBuilder = new TermQueryBuilder(TEXT_FIELD_NAME, "bar");
        WrapperQueryBuilder queryBuilder = new WrapperQueryBuilder(termQueryBuilder.toString());
        UnsupportedOperationException e = expectThrows(
            UnsupportedOperationException.class,
            () -> queryBuilder.toQuery(createSearchExecutionContext())
        );
        assertThat(e.getMessage(), Matchers.equalTo("this query must be rewritten first"));

        QueryBuilder rewrite = queryBuilder.rewrite(createSearchExecutionContext());
        assertThat(rewrite, Matchers.equalTo(termQueryBuilder));
    }

    /**
     * Tests that the query is rewritten with an inner name as expected.
     */
    @Test
    public void testRewriteWithInnerName() throws IOException {
        QueryBuilder builder = new WrapperQueryBuilder("""
            { "match_all" : {"_name" : "foobar"}}""");
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();
        assertThat(builder.rewrite(searchExecutionContext), Matchers.equalTo(new MatchAllQueryBuilder().queryName("foobar")));

        builder = new WrapperQueryBuilder("""
            { "match_all" : {"_name" : "foobar"}}""").queryName("outer");
        assertThat(
            builder.rewrite(searchExecutionContext),
            Matchers.equalTo(new BoolQueryBuilder().must(new MatchAllQueryBuilder().queryName("foobar")).queryName("outer"))
        );
    }

    /**
     * Tests that the query is rewritten with an inner boost as expected.
     */
    @Test
    public void testRewriteWithInnerBoost() throws IOException {
        final TermQueryBuilder termQueryBuilder = new TermQueryBuilder(TEXT_FIELD_NAME, "bar").boost(2);
        QueryBuilder builder = new WrapperQueryBuilder(termQueryBuilder.toString());
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();
        assertThat(builder.rewrite(searchExecutionContext), Matchers.equalTo(termQueryBuilder));

        builder = new WrapperQueryBuilder(termQueryBuilder.toString()).boost(3);
        assertThat(
            builder.rewrite(searchExecutionContext),
            Matchers.equalTo(new BoolQueryBuilder().must(termQueryBuilder).boost(3))
        );
    }

    /**
     * Tests that the inner query is also rewritten as expected.
     */
    @Test
    public void testRewriteInnerQueryToo() throws IOException {
        SearchExecutionContext searchExecutionContext = createSearchExecutionContext();

        QueryBuilder queryBuilder = new WrapperQueryBuilder(
            new WrapperQueryBuilder(new TermQueryBuilder(TEXT_FIELD_NAME, "bar").toString()).toString()
        );
        assertThat(queryBuilder.rewrite(searchExecutionContext).toQuery(searchExecutionContext), Matchers.equalTo(new TermQuery(new Term(TEXT_FIELD_NAME, "bar"))));

        queryBuilder = new WrapperQueryBuilder(
            new WrapperQueryBuilder(new WrapperQueryBuilder(new TermQueryBuilder(TEXT_FIELD_NAME, "bar").toString()).toString()).toString()
        );
        assertThat(queryBuilder.rewrite(searchExecutionContext).toQuery(searchExecutionContext), Matchers.equalTo(new TermQuery(new Term(TEXT_FIELD_NAME, "bar"))));

        queryBuilder = new WrapperQueryBuilder(new BoolQueryBuilder().toString());
        assertThat(queryBuilder.rewrite(searchExecutionContext).toQuery(searchExecutionContext), Matchers.equalTo(new MatchAllDocsQuery()));
    }

    /**
     * Tests that the maximum nested depth is enforced.
     */
    @Test
    public void testMaxNestedDepth() throws IOException {
        BoolQueryBuilderTests boolQueryBuilderTests = new BoolQueryBuilderTests();
        BoolQueryBuilder boolQuery = boolQueryBuilderTests.createQueryWithInnerQuery(new MatchAllQueryBuilder());
        int maxDepth = randomIntBetween(3, 5);
        AbstractQueryBuilder.setMaxNestedDepth(maxDepth);
        for (int i = 1; i < maxDepth - 1; i++) {
            boolQuery = boolQueryBuilderTests.createQueryWithInnerQuery(boolQuery);
        }
        WrapperQueryBuilder query = new WrapperQueryBuilder(Strings.toString(boolQuery));
        AbstractQueryBuilder.setMaxNestedDepth(maxDepth);
        try {
            // no errors, we reached the limit but we did not go beyond it
            query.rewrite(createSearchExecutionContext());
            // one more level causes an exception
            WrapperQueryBuilder q = new WrapperQueryBuilder(Strings.toString(boolQueryBuilderTests.createQueryWithInnerQuery(boolQuery)));
            IllegalArgumentException e = expectThrows(XContentParseException.class, () -> q.rewrite(createSearchExecutionContext()));
            // there may be nested XContentParseExceptions coming from ObjectParser, we just extract the root cause
            while (e.getCause() != null) {
                assertThat(e.getCause(), Matchers.instanceOf(IllegalArgumentException.class));
                e = (IllegalArgumentException) e.getCause();
            }

            assertThat(
                e.getMessage(),
                Matchers.equalTo(
                    String.format(
                        "The nested depth of the query exceeds the maximum nested depth for queries set in [%s]",
                        INDICES_MAX_NESTED_DEPTH_SETTING.getKey()
                    )
                )
            );
        } finally {
            AbstractQueryBuilder.setMaxNestedDepth(INDICES_MAX_NESTED_DEPTH_SETTING.getDefault(Settings.EMPTY));
        }
    }
}