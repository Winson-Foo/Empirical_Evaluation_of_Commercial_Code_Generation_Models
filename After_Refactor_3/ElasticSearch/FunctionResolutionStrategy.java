package org.elasticsearch.xpack.ql.expression.function;

import org.elasticsearch.xpack.ql.session.Configuration;

/**
 * Strategy indicating the type of resolution to apply for resolving the actual function definition in a pluggable way.
 */
public interface FunctionResolutionStrategy {

    /**
     * Build the real function from this one and resolution metadata.
     *
     * @param uf the unresolved function
     * @param cfg the configuration
     * @param def the function definition
     * @return the resolved function
     */
    Function buildResolved(UnresolvedFunction uf, Configuration cfg, FunctionDefinition def);

    /**
     * The kind of strategy being applied. Used when building the error message sent back to the user when
     * they specify a function that doesn't exist.
     *
     * @return the kind of strategy being applied
     */
    String kind();

    /**
     * Is {@code def} a valid alternative for function invocations
     * of this kind. Used to filter the list of "did you mean"
     * options sent back to the user when they specify a missing function.
     *
     * @param def the function definition to check
     * @return true if this definition is a valid alternative
     */
    boolean isValidAlternative(FunctionDefinition def);

    /**
     * Default behavior of standard function calls like {@code ABS(col)}.
     */
    final class Default implements FunctionResolutionStrategy {

        @Override
        public Function buildResolved(UnresolvedFunction uf, Configuration cfg, FunctionDefinition def) {
            return def.builder().build(uf, cfg);
        }

        @Override
        public String kind() {
            return "function";
        }

        @Override
        public boolean isValidAlternative(FunctionDefinition def) {
            return true;
        }
    }

}

