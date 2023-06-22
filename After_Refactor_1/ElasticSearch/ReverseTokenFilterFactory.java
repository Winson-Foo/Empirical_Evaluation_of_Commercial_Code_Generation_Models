package org.elasticsearch.analysis.common;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.reverse.ReverseStringFilter;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.analysis.AbstractTokenFilterFactory;

/**
 * A token filter factory to reverse the tokens in a stream.
 */
public class ReverseTokenFilterFactory extends AbstractTokenFilterFactory {

    /**
     * Constructor.
     *
     * @param indexSettings the index settings
     * @param environment   the environment
     * @param name          the name of the filter
     * @param settings      the settings for the filter
     */
    public ReverseTokenFilterFactory(final IndexSettings indexSettings, final Environment environment,
                                     final String name, final Settings settings) {
        super(name, settings);
    }

    /**
     * Create a token stream with a reverse filter applied.
     *
     * @param tokenStream the original token stream
     * @return the token stream with reverse filter
     */
    @Override
    public TokenStream create(final TokenStream tokenStream) {
        return new ReverseStringFilter(tokenStream);
    }
}