/**
 * This package contains classes and interfaces related to functions used in the Elasticsearch Query Language (QL).
 */
package org.elasticsearch.xpack.ql.expression.function;

/**
 * Marker interface indicating that a function accepts two optional arguments (the last two).
 * This is used by the {@link FunctionRegistry} to perform validation of function declaration.
 */
public interface TwoOptionalArguments {

    // No additional methods beyond marker interface
    
}

// Other related classes and interfaces could be included in this package, such as:
// - Function: An interface that all functions in QL must implement
// - FunctionRegistry: A registry for registering and validating QL functions
// - FunctionFactory: An interface for creating new instances of a specified QL function based on input arguments
// - FunctionResult: An interface for representing the result of a QL function call
// - FunctionContext: A context object that holds information needed for evaluating QL function expressions
// - AbstractFunction: A base implementation of the Function interface that includes common functionality and error handling code.

