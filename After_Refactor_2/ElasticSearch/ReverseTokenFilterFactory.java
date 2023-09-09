// ReverseTokenFilterFactory creates a token filter that reverses the order of characters in tokens.
public class ReverseTokenFilterFactory extends AbstractTokenFilterFactory {

    public ReverseTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
        super(name, settings);
    }

    // create creates the reverse string filter token stream.
    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new ReverseStringFilter(tokenStream);
    }
}