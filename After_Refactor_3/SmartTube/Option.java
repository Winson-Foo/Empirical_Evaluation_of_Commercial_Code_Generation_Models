package com.liskovsoft.smartyoutubetv2.common.app.models.search.vineyard;

public class Option {

    private String title;
    private String value;
    private int iconResource;

    public Option(String title, String value, int iconResource) {
        setTitle(title);
        setValue(value);
        setIconResource(iconResource);
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public int getIconResource() {
        return iconResource;
    }

    public void setIconResource(int iconResource) {
        this.iconResource = iconResource;
    }
}

