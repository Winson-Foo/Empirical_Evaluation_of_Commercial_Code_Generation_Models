package org.elasticsearch.xpack.ml.extractor;

import org.elasticsearch.common.document.DocumentField;
import org.elasticsearch.search.SearchHit;

import java.util.List;
import java.util.Objects;
import java.util.Set;

abstract class AbstractField implements ExtractedField {

    private final String name;
    private final Set<String> types;

    AbstractField(String name, Set<String> types) {
        this.name = Objects.requireNonNull(name);
        this.types = Objects.requireNonNull(types);
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public String getSearchField() {
        return name;
    }

    @Override
    public Set<String> getTypes() {
        return types;
    }

    protected Object[] getFieldValue(SearchHit hit) {
        DocumentField keyValue = hit.field(getSearchField());

        if (keyValue != null) {
            List<Object> values = keyValue.getValues();
            return values.toArray(new Object[values.size()]);
        }

        return new Object[0];
    }
}