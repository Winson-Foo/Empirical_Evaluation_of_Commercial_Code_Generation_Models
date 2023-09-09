package com.liskovsoft.smartyoutubetv2.common.app.presenters;

import android.content.Context;
import com.liskovsoft.smartyoutubetv2.common.app.models.playback.ui.OptionItem;
import com.liskovsoft.smartyoutubetv2.common.app.views.AppDialogView;

import java.util.List;

public class AppDialogPresenter extends BasePresenter<AppDialogView> {

    private final AppDialogViewManager dialogManager;

    public AppDialogPresenter(Context context, AppDialogViewManager dialogManager) {
        super(context);
        this.dialogManager = dialogManager;
    }

    public void showDialog(String title, List<OptionCategory> categories) {
        dialogManager.showDialog(title, categories);
    }

    public void closeDialog() {
        dialogManager.closeDialog();
    }

    public boolean isDialogShown() {
        return dialogManager.isDialogShown();
    }

    public void appendStringList(String categoryTitle, List<OptionItem> items) {
        dialogManager.appendStringList(categoryTitle, items);
    }

    public void appendRadioList(String categoryTitle, List<OptionItem> items) {
        dialogManager.appendRadioList(categoryTitle, items);
    }

    public void appendCheckBoxList(String categoryTitle, List<OptionItem> items) {
        dialogManager.appendCheckBoxList(categoryTitle, items);
    }

    public void appendLongText(String categoryTitle, OptionItem item) {
        dialogManager.appendLongText(categoryTitle, item);
    }

    public void appendChat(String categoryTitle, OptionItem item) {
        dialogManager.appendChat(categoryTitle, item);
    }

    public void appendComments(String categoryTitle, OptionItem item) {
        dialogManager.appendComments(categoryTitle, item);
    }

    public void appendSingleSwitch(OptionItem optionItem) {
        dialogManager.appendSingleSwitch(optionItem);
    }

    public void appendSingleButton(OptionItem optionItem) {
        dialogManager.appendSingleButton(optionItem);
    }

    public void showMessage(String title, long timeoutMs) {
        dialogManager.showMessage(title, timeoutMs);
    }

    public void setCloseTimeoutMs(long timeoutMs) {
        dialogManager.setCloseTimeoutMs(timeoutMs);
    }

    public void enableTransparent(boolean enable) {
        dialogManager.enableTransparent(enable);
    }

    public boolean isTransparent() {
        return dialogManager.isTransparent();
    }

    public void enableExpandable(boolean enable) {
        dialogManager.enableExpandable(enable);
    }

    public boolean isEmpty() {
        return dialogManager.isEmpty();
    }
} 