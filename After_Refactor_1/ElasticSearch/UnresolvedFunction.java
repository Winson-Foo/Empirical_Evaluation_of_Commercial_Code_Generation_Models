package org.elasticsearch.xpack.ql.expression.function;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import org.elasticsearch.xpack.ql.capabilities.Unresolvable;
import org.elasticsearch.xpack.ql.capabilities.UnresolvedException;
import org.elasticsearch.xpack.ql.expression.Expression;
import org.elasticsearch.xpack.ql.expression.Nullability;
import org.elasticsearch.xpack.ql.expression.gen.script.ScriptTemplate;
import org.elasticsearch.xpack.ql.session.Configuration;
import org.elasticsearch.xpack.ql.tree.NodeInfo;
import org.elasticsearch.xpack.ql.tree.Source;
import org.elasticsearch.xpack.ql.type.DataType;
import org.elasticsearch.xpack.ql.util.StringUtils;

/**
 * Represents an unresolved function.
 */
public class UnresolvedFunction extends Function implements Unresolvable {

    private final String functionName;
    private final String unresolvedMessage;
    private final FunctionResolutionStrategy resolutionStrategy;

    /**
     * Flag to indicate analysis has been applied and there's no point in
     * doing it again this is an optimization to prevent searching for a
     * better unresolved message over and over again.
     */
    private final boolean hasBeenAnalyzed;

    public UnresolvedFunction(
            Source source,
            String functionName,
            FunctionResolutionStrategy resolutionStrategy,
            List<Expression> arguments) {
        this(source, functionName, resolutionStrategy, arguments, false, null);
    }

    /**
     * Constructor used for specifying a more descriptive message (typically
     * 'did you mean') instead of the default one.
     *
     * @see #withMessage(String)
     */
    public UnresolvedFunction(
            Source source,
            String functionName,
            FunctionResolutionStrategy resolutionStrategy,
            List<Expression> arguments,
            boolean hasBeenAnalyzed,
            String unresolvedMessage) {
        super(source, arguments);
        this.functionName = functionName;
        this.resolutionStrategy = resolutionStrategy;
        this.hasBeenAnalyzed = hasBeenAnalyzed;
        this.unresolvedMessage = unresolvedMessage == null ?
                "Unknown " + resolutionStrategy.kind() + " [" + functionName + "]" :
                unresolvedMessage;
    }

    @Override
    protected NodeInfo<UnresolvedFunction> info() {
        return NodeInfo.create(
                this,
                UnresolvedFunction::new,
                functionName,
                resolutionStrategy,
                arguments(),
                hasBeenAnalyzed,
                unresolvedMessage);
    }

    @Override
    public Expression replaceChildren(List<Expression> newArguments) {
        return new UnresolvedFunction(
                source(),
                functionName,
                resolutionStrategy,
                newArguments,
                hasBeenAnalyzed,
                unresolvedMessage);
    }

    /**
     * Builds a function to replace this one after resolving the function.
     *
     * @param configuration The configuration for the session.
     * @param functionDef The resolved function definition.
     * @return The resolved function.
     */
    public Function buildResolved(Configuration configuration, FunctionDefinition functionDef) {
        return resolutionStrategy.buildResolved(this, configuration, functionDef);
    }

    /**
     * Builds a marker {@link UnresolvedFunction} with an error message
     * about the function being missing.
     *
     * @param normalizedFunctionName The normalized function name being searched for.
     * @param functionDefs The available function definitions.
     * @return The unresolved function with an error message suggesting an alternative.
     */
    public UnresolvedFunction missing(String normalizedFunctionName, Iterable<FunctionDefinition> functionDefs) {
        Set<String> functionNames = new LinkedHashSet<>();
        for (FunctionDefinition def : functionDefs) {
            if (resolutionStrategy.isValidAlternative(def)) {
                functionNames.add(def.name());
                functionNames.addAll(def.aliases());
            }
        }
        List<String> matches = StringUtils.findSimilar(normalizedFunctionName, functionNames);
        if (matches.isEmpty()) {
            return this;
        }
        String matchesMessage = matches.size() == 1 ? "[" + matches.get(0) + "]" : "any of " + matches;
        return withMessage("Unknown " + resolutionStrategy.kind() + " [" + functionName + "], did you mean " + matchesMessage + "?");
    }

    public UnresolvedFunction withMessage(String message) {
        return new UnresolvedFunction(
                source(),
                functionName,
                resolutionStrategy,
                arguments(),
                true,
                message);
    }

    @Override
    public boolean resolved() {
        return false;
    }

    public String name() {
        return functionName;
    }

    public FunctionResolutionStrategy resolutionStrategy() {
        return resolutionStrategy;
    }

    public boolean hasBeenAnalyzed() {
        return hasBeenAnalyzed;
    }

    @Override
    public DataType dataType() {
        throw new UnresolvedException("dataType", this);
    }

    @Override
    public Nullability nullable() {
        throw new UnresolvedException("nullable", this);
    }

    @Override
    public ScriptTemplate asScript() {
        throw new UnresolvedException("script", this);
    }

    @Override
    public String unresolvedMessage() {
        return unresolvedMessage;
    }

    @Override
    public String toString() {
        return UNRESOLVED_PREFIX + functionName + arguments();
    }

    @Override
    public String nodeString() {
        return toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null || obj.getClass() != getClass()) {
            return false;
        }
        UnresolvedFunction other = (UnresolvedFunction) obj;
        return functionName.equals(other.functionName)
                && resolutionStrategy.equals(other.resolutionStrategy)
                && arguments().equals(other.arguments())
                && hasBeenAnalyzed == other.hasBeenAnalyzed
                && Objects.equals(unresolvedMessage, other.unresolvedMessage);
    }

    @Override
    public int hashCode() {
        return Objects.hash(functionName, resolutionStrategy, arguments(), hasBeenAnalyzed, unresolvedMessage);
    }
}