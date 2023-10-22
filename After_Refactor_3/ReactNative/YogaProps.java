package com.facebook.yoga;

public interface YogaDimensionProps {
  void setWidth(float width);
  void setWidthPercent(float percent);
  void setMinWidth(float minWidth);
  void setMinWidthPercent(float percent);
  void setMaxWidth(float maxWidth);
  void setMaxWidthPercent(float percent);
  void setWidthAuto();

  void setHeight(float height);
  void setHeightPercent(float percent);
  void setMinHeight(float minHeight);
  void setMinHeightPercent(float percent);
  void setMaxHeight(float maxHeight);
  void setMaxHeightPercent(float percent);
  void setHeightAuto();

  YogaValue getWidth();
  YogaValue getMinWidth();
  YogaValue getMaxWidth();
  YogaValue getHeight();
  YogaValue getMinHeight();
  YogaValue getMaxHeight();
}

public interface YogaMarginProps {
  void setMargin(YogaEdge edge, float margin);
  void setMarginPercent(YogaEdge edge, float percent);
  void setMarginAuto(YogaEdge edge);
  YogaValue getMargin(YogaEdge edge);
}

public interface YogaPaddingProps {
  void setPadding(YogaEdge edge, float padding);
  void setPaddingPercent(YogaEdge edge, float percent);
  YogaValue getPadding(YogaEdge edge);
}

public interface YogaPositionProps {
  void setPositionType(YogaPositionType positionType);
  void setPosition(YogaEdge edge, float position);
  void setPositionPercent(YogaEdge edge, float percent);
  YogaValue getPosition(YogaEdge edge);
}

public interface YogaFlexProps {
  void setFlex(float flex);
  void setFlexBasisAuto();
  void setFlexBasisPercent(float percent);
  void setFlexBasis(float flexBasis);
  void setFlexDirection(YogaFlexDirection direction);
  void setFlexGrow(float flexGrow);
  void setFlexShrink(float flexShrink);

  float getFlexGrow();
  float getFlexShrink();
  YogaValue getFlexBasis();
}

public interface YogaAlignProps {
  void setAlignContent(YogaAlign alignContent);
  void setAlignItems(YogaAlign alignItems);
  void setAlignSelf(YogaAlign alignSelf);

  YogaAlign getAlignItems();
  YogaAlign getAlignSelf();
  YogaAlign getAlignContent();
}

public interface YogaOtherProps {
  void setJustifyContent(YogaJustify justifyContent);
  void setDirection(YogaDirection direction);
  void setBorder(YogaEdge edge, float value);
  void setWrap(YogaWrap wrap);
  void setAspectRatio(float aspectRatio);
  void setIsReferenceBaseline(boolean isReferenceBaseline);
  void setMeasureFunction(YogaMeasureFunction measureFunction);
  void setBaselineFunction(YogaBaselineFunction yogaBaselineFunction);

  YogaDirection getStyleDirection();
  YogaFlexDirection getFlexDirection();
  YogaJustify getJustifyContent();
  YogaWrap getWrap();
  float getAspectRatio();
  boolean getIsReferenceBaseline();
  YogaMeasureFunction getMeasureFunction();
  YogaBaselineFunction getBaselineFunction();
}

public interface YogaProps extends
    YogaDimensionProps,
    YogaMarginProps,
    YogaPaddingProps,
    YogaPositionProps,
    YogaFlexProps,
    YogaAlignProps,
    YogaOtherProps {
  // empty interface
} 