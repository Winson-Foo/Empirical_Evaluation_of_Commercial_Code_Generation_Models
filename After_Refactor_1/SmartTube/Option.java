package com.liskovsoft.smartyoutubetv2.common.app.models.search.vineyard;

public class SearchOption {
    private final String title;
    private final String value;
    private final int iconResource;

    public SearchOption(String title, String value, int iconResource) {
        this.title = title;
        this.value = value;
        this.iconResource = iconResource;
    }

    public String getTitle() {
        return title;
    }

    public String getValue() {
        return value;
    }

    public int getIconResource() {
        return iconResource;
    }
}

