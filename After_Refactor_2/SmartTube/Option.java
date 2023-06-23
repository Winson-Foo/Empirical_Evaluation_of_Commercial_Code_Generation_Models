package com.liskovsoft.smartyoutubetv2.common.app.models.search.vineyard;

public class Option {
    private String mTitle;
    private String mValue;
    private int mIconResource;

    public Option(String title, String value, int iconResource) {
        mTitle = title;
        mValue = value;
        mIconResource = iconResource;
    }

    public String getTitle() {
        return mTitle;
    }

    public void setTitle(String title) {
        mTitle = title;
    }

    public String getValue() {
        return mValue;
    }

    public void setValue(String value) {
        mValue = value;
    }

    public int getIconResource() {
        return mIconResource;
    }

    public void setIconResource(int iconResource) {
        mIconResource = iconResource;
    }
}

