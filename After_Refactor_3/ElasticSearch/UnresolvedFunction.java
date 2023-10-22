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

public class UnresolvedFunction extends Function implements Unresolvable {

    private final String functionName;
    private final String unresolvedMessage;
    private final FunctionResolutionStrategy resolutionStrategy;
    private final boolean analyzed;

    public UnresolvedFunction(Source source, String functionName, FunctionResolutionStrategy resolutionStrategy, List<Expression> children) {
        super(source, children);
        this.functionName = functionName;
        this.resolutionStrategy = resolutionStrategy;
        this.analyzed = false;
        this.unresolvedMessage = "Unknown " + resolutionStrategy.kind() + " [" + functionName + "]";
    }

    UnresolvedFunction(
        Source source,
        String functionName,
        FunctionResolutionStrategy resolutionStrategy,
        List<Expression> children,
        boolean analyzed,
        String unresolvedMessage
    ) {
        super(source, children);
        this.functionName = functionName;
        this.resolutionStrategy = resolutionStrategy;
        this.analyzed = analyzed;
        this.unresolvedMessage = unresolvedMessage == null ? "Unknown " + resolutionStrategy.kind() + " [" + functionName + "]" : unresolvedMessage;
    }

    public UnresolvedFunction withMessage(String message) {
        return new UnresolvedFunction(source(), functionName, resolutionStrategy, children(), true, message);
    }

    public Function buildResolved(Configuration configuration, FunctionDefinition def) {
        return resolutionStrategy.buildResolved(this, configuration, def);
    }

    public UnresolvedFunction missing(String normalizedName, Iterable<FunctionDefinition> alternatives) {
        Set<String> names = new LinkedHashSet<>();
        for (FunctionDefinition def : alternatives) {
            if (resolutionStrategy.isValidAlternative(def)) {
                names.add(def.name());
                names.addAll(def.aliases());
            }
        }

        List<String> matches = StringUtils.findSimilar(normalizedName, names);
        if (matches.isEmpty()) {
            return this;
        }
        String matchesMessage = matches.size() == 1 ? "[" + matches.get(0) + "]" : "any of " + matches;

        return withMessage("Unknown " + resolutionStrategy.kind() + " [" + functionName + "], did you mean " + matchesMessage + "?");
    }

    @Override
    public boolean resolved() {
        return false;
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
    protected NodeInfo<UnresolvedFunction> info() {
        return NodeInfo.create(this, UnresolvedFunction::new, functionName, resolutionStrategy, children(), analyzed, unresolvedMessage);
    }

    @Override
    public Expression replaceChildren(List<Expression> newChildren) {
        return new UnresolvedFunction(source(), functionName, resolutionStrategy, newChildren, analyzed, unresolvedMessage);
    }

    public String getFunctionName() {
        return functionName;
    }

    public FunctionResolutionStrategy getResolutionStrategy() {
        return resolutionStrategy;
    }

    public boolean isAnalyzed() {
        return analyzed;
    }

    @Override
    public String toString() {
        return UNRESOLVED_PREFIX + functionName + children();
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
        return functionName.equals(other.functionName) &&
            resolutionStrategy.equals(other.resolutionStrategy) &&
            children().equals(other.children()) &&
            analyzed == other.analyzed &&
            Objects.equals(unresolvedMessage, other.unresolvedMessage);
    }

    @Override
    public int hashCode() {
        return Objects.hash(functionName, resolutionStrategy, children(), analyzed, unresolvedMessage);
    }
}