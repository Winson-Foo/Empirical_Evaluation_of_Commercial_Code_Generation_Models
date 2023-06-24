package org.elasticsearch.xpack.ml.extractor;

import org.elasticsearch.common.document.DocumentField;
import org.elasticsearch.search.SearchHit;

import java.util.List;
import java.util.Set;

/**
 * A SearchFieldExtractor provides a method for retrieving the values of a specific field in a SearchHit.
 */
abstract class SearchFieldExtractor implements ExtractedField {

    private final String name;
    private final Set<String> types;

    public SearchFieldExtractor(String name, Set<String> types) {
        this.name = name;
        this.types = types;
    }

    @Override
    public String getName() {
        return name;
    }

    /**
     * Returns the name of the field that should be searched for in the SearchHit.
     * By default, this is the same as the ExtractedField's name.
     */
    @Override
    public String getSearchField() {
        return getName();
    }

    @Override
    public Set<String> getTypes() {
        return types;
    }

    /**
     * Retrieves the values of the SearchFieldExtractor's field from the given SearchHit.
     */
    protected Object[] getFieldValue(SearchHit hit) {
        DocumentField keyValue = hit.getField(getSearchField());
        if (keyValue != null) {
            List<Object> values = keyValue.getValues();
            return values.toArray(new Object[values.size()]);
        }
        return new Object[0];
    }
}