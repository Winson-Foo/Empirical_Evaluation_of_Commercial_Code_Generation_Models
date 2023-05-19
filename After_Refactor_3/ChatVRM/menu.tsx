import React, { useContext, useRef, useState } from "react";
import PropTypes from "prop-types";
import { IconButton } from "./iconButton";
import { Message } from "@/features/messages/messages";
import { KoeiroParam } from "@/features/constants/koeiroParam";
import { ChatLog } from "./chatLog";
import { Settings } from "./settings";
import { ViewerContext } from "@/features/vrmViewer/viewerContext";
import { AssistantText } from "./assistantText";

function MenuButton({ iconName, label, isProcessing, onClick }) {
  return (
    <IconButton
      iconName={iconName}
      label={label}
      isProcessing={isProcessing}
      onClick={onClick}
    />
  );
}

function Menu({ openAiKey, systemPrompt, chatLog, koeiroParam, assistantMessage,
  onChangeSystemPrompt, onChangeAiKey, onChangeChatLog, onChangeKoeiromapParam }) {

  const { viewer } = useContext(ViewerContext);
  const fileInputRef = useRef(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showChatLog, setShowChatLog] = useState(false);

  const handleChangeSystemPrompt = (event) => {
    onChangeSystemPrompt(event.target.value);
  };

  const handleAiKeyChange = (event) => {
    onChangeAiKey(event.target.value);
  };

  const handleChangeKoeiroParam = (x, y) => {
    onChangeKoeiromapParam({
      speakerX: x,
      speakerY: y,
    });
  };

  const handleClickOpenVrmFile = () => {
    fileInputRef.current?.click();
  };

  const handleChangeVrmFile = (event) => {
    const files = event.target.files;
    if (!files) return;

    const file = files[0];
    if (!file) return;

    const file_type = file.name.split(".").pop();

    if (file_type === "vrm") {
      const blob = new Blob([file], { type: "application/octet-stream" });
      const url = window.URL.createObjectURL(blob);
      viewer.loadVrm(url);
    }

    event.target.value = "";
  };

  return (
    <>
      <div className="absolute z-10 m-24">
        <div className="grid grid-flow-col gap-[8px]">
          <MenuButton
            iconName="24/Menu"
            label="設定"
            isProcessing={false}
            onClick={() => setShowSettings(true)}
          />
          {showChatLog ? (
            <MenuButton
              iconName="24/CommentOutline"
              label="会話ログ"
              isProcessing={false}
              onClick={() => setShowChatLog(false)}
            />
          ) : (
            <MenuButton
              iconName="24/CommentFill"
              label="会話ログ"
              isProcessing={false}
              disabled={chatLog.length <= 0}
              onClick={() => setShowChatLog(true)}
            />
          )}
        </div>
      </div>
      {showChatLog && <ChatLog messages={chatLog} />}
      {showSettings && (
        <Settings
          openAiKey={openAiKey}
          chatLog={chatLog}
          systemPrompt={systemPrompt}
          koeiroParam={koeiroParam}
          onClickClose={() => setShowSettings(false)}
          onChangeAiKey={handleAiKeyChange}
          onChangeSystemPrompt={handleChangeSystemPrompt}
          onChangeChatLog={onChangeChatLog}
          onChangeKoeiroParam={handleChangeKoeiroParam}
          onClickOpenVrmFile={handleClickOpenVrmFile}
        />
      )}
      {!showChatLog && assistantMessage && (
        <AssistantText message={assistantMessage} />
      )}
      <input
        type="file"
        className="hidden"
        accept=".vrm"
        ref={fileInputRef}
        onChange={handleChangeVrmFile}
      />
    </>
  );
}

Menu.propTypes = {
  openAiKey: PropTypes.string,
  systemPrompt: PropTypes.string,
  chatLog: PropTypes.arrayOf(PropTypes.object),
  koeiroParam: PropTypes.shape({
    speakerX: PropTypes.number,
    speakerY: PropTypes.number,
  }),
  assistantMessage: PropTypes.string,
  onChangeSystemPrompt: PropTypes.func,
  onChangeAiKey: PropTypes.func,
  onChangeChatLog: PropTypes.func,
  onChangeKoeiromapParam: PropTypes.func,
};

export default Menu;

