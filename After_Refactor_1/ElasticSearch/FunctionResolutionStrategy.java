/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ql.expression.function;

import org.elasticsearch.xpack.ql.session.Configuration;

/**
 * Strategy indicating the type of resolution to apply for resolving the actual function definition in a pluggable way.
 */
public interface FunctionResolutionStrategy {

    /**
     * Default behavior of standard function calls like {@code ABS(col)}.
     */
    FunctionResolutionStrategy DEFAULT = new DefaultFunctionResolutionStrategy();

    /**
     * Build the real function from this one and resolution metadata.
     */
    Function buildResolved(UnresolvedFunction uf, Configuration cfg, FunctionDefinition def);

    /**
     * The kind of strategy being applied. Used when
     * building the error message sent back to the user when
     * they specify a function that doesn't exist.
     */
    String kind();

    /**
     * Is {@code def} a valid alternative for function invocations
     * of this kind. Used to filter the list of "did you mean"
     * options sent back to the user when they specify a missing
     * function.
     */
    boolean isValidAlternative(FunctionDefinition def);

    /**
     * Default function resolution strategy.
     */
    class DefaultFunctionResolutionStrategy implements FunctionResolutionStrategy {
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