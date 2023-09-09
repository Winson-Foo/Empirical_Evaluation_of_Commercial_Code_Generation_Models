/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.facebook.yoga;

public interface WidthProps {
  void setWidth(float width);
  void setWidthPercent(float percent);
  void setMinWidth(float minWidth);
  void setMinWidthPercent(float percent);
  void setMaxWidth(float maxWidth);
  void setMaxWidthPercent(float percent);
  void setWidthAuto();
  YogaValue getWidth();
  YogaValue getMinWidth();
  YogaValue getMaxWidth();
}

public interface HeightProps {
  void setHeight(float height);
  void setHeightPercent(float percent);
  void setMinHeight(float minHeight);
  void setMinHeightPercent(float percent);
  void setMaxHeight(float maxHeight);
  void setMaxHeightPercent(float percent);
  void setHeightAuto();
  YogaValue getHeight();
  YogaValue getMinHeight();
  YogaValue getMaxHeight();
}

public interface MarginProps {
  void setMargin(YogaEdge edge, float margin);
  void setMarginPercent(YogaEdge edge, float percent);
  void setMarginAuto(YogaEdge edge);
  YogaValue getMargin(YogaEdge edge);
}

public interface PaddingProps {
  void setPadding(YogaEdge edge, float padding);
  void setPaddingPercent(YogaEdge edge, float percent);
  YogaValue getPadding(YogaEdge edge);
}

public interface PositionProps {
  void setPositionType(YogaPositionType positionType);
  void setPosition(YogaEdge edge, float position);
  void setPositionPercent(YogaEdge edge, float percent);
  YogaPositionType getPositionType();
  YogaValue getPosition(YogaEdge edge);
}

public interface AlignmentProps {
  void setAlignContent(YogaAlign alignContent);
  void setAlignItems(YogaAlign alignItems);
  void setAlignSelf(YogaAlign alignSelf);
  YogaAlign getAlignContent();
  YogaAlign getAlignItems();
  YogaAlign getAlignSelf();
}

public interface FlexProps {
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

public interface OtherProps {
  void setJustifyContent(YogaJustify justifyContent);
  void setDirection(YogaDirection direction);
  void setBorder(YogaEdge edge, float value);
  void setWrap(YogaWrap wrap);
  void setAspectRatio(float aspectRatio);
  void setIsReferenceBaseline(boolean isReferenceBaseline);
  void setMeasureFunction(YogaMeasureFunction measureFunction);
  void setBaselineFunction(YogaBaselineFunction yogaBaselineFunction);
  YogaJustify getJustifyContent();
  YogaDirection getStyleDirection();
  float getAspectRatio();
  boolean getIsReferenceBaseline();
  YogaWrap getWrap();
  YogaBaselineFunction getBaselineFunction();
  YogaMeasureFunction getMeasureFunction();
}

public interface YogaProps extends WidthProps, HeightProps, MarginProps, PaddingProps, PositionProps,
    AlignmentProps, FlexProps, OtherProps {
  // Empty interface to combine all other property interfaces
  // This is the main interface that clients should use to set and get properties.
}