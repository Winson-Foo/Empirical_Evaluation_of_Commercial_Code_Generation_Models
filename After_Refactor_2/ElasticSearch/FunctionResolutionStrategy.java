package org.elasticsearch.xpack.ql.expression.function;

import org.elasticsearch.xpack.ql.session.Configuration;

/**
 * Interface for resolving the definition of a function in a pluggable way.
 */
public interface FunctionResolver {

    /**
     * Default implementation for resolving function definitions.
     */
    FunctionResolver DEFAULT = new DefaultFunctionResolver();

    /**
     * Build the real function from an unresolved function and resolution metadata.
     */
    Function buildResolved(UnresolvedFunction unresolvedFunction, Configuration configuration, FunctionDefinition functionDefinition);

    /**
     * Returns the type of resolution strategy being used - used in error messages.
     */
    String getResolutionType();

    /**
     * Determines whether the function definition is a valid alternative - used when suggesting alternatives
     * to the user when they specify a nonexistent function.
     */
    boolean isValidAlternative(FunctionDefinition functionDefinition);

    /**
     * Default implementation for FunctionResolver.
     */
    class DefaultFunctionResolver implements FunctionResolver {
        @Override
        public Function buildResolved(UnresolvedFunction unresolvedFunction, Configuration configuration, FunctionDefinition functionDefinition) {
            return functionDefinition.builder().build(unresolvedFunction, configuration);
        }

        @Override
        public String getResolutionType() {
            return "function";
        }

        @Override
        public boolean isValidAlternative(FunctionDefinition functionDefinition) {
            return true;
        }
    }
}