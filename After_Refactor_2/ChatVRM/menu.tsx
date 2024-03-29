import { IconButton } from "./iconButton";
import { Message } from "@/features/messages/messages";
import { KoeiroParam } from "@/features/constants/koeiroParam";
import { ChatLog } from "./chatLog";
import React, { useCallback, useContext, useRef, useState } from "react";
import { Settings } from "./settings";
import { ViewerContext } from "@/features/vrmViewer/viewerContext";
import { AssistantText } from "./assistantText";

interface Props { // Use interface instead of type
  openAiKey: string;
  systemPrompt: string;
  chatLog: Message[];
  koeiroParam: KoeiroParam;
  assistantMessage: string;
  onChangeSystemPrompt: (systemPrompt: string) => void;
  onChangeAiKey: (key: string) => void;
  onChangeChatLog: (index: number, text: string) => void;
  onChangeKoeiromapParam: (param: KoeiroParam) => void;
}

const Menu: React.FC<Props> = ({ // Use React.FC instead of arrow function
  openAiKey,
  systemPrompt,
  chatLog,
  koeiroParam,
  assistantMessage,
  onChangeSystemPrompt,
  onChangeAiKey,
  onChangeChatLog,
  onChangeKoeiromapParam,
}) => {
  const [showSettings, setShowSettings] = useState(false);
  const [showChatLog, setShowChatLog] = useState(false);
  const { viewer } = useContext(ViewerContext);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChangeSystemPrompt = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChangeSystemPrompt(event.target.value);
  }, [onChangeSystemPrompt]);

  const handleAiKeyChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    onChangeAiKey(event.target.value);
  }, [onChangeAiKey]);

  const handleChangeKoeiroParam = useCallback((x: number, y: number) => {
    onChangeKoeiromapParam({ speakerX: x, speakerY: y });
  }, [onChangeKoeiromapParam]);

  const handleClickOpenVrmFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleChangeVrmFile = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const file = files[0];
    if (!file) return;

    const fileType = file.name.split(".").pop();
    if (fileType === "vrm") {
      const blob = new Blob([file], { type: "application/octet-stream" });
      const url = window.URL.createObjectURL(blob);
      viewer.loadVrm(url);
    }
    event.target.value = "";
  }, [viewer]);

  const handleToggleSettings = useCallback(() => {
    setShowSettings(!showSettings);
  }, [showSettings]);

  const handleToggleChatLog = useCallback(() => {
    setShowChatLog(!showChatLog);
  }, [showChatLog]);

  const renderButtons = () => (
    <div className="grid grid-flow-col gap-[8px]">
      <IconButton
        iconName="24/Menu"
        label="設定"
        isProcessing={false}
        onClick={handleToggleSettings}
      />
      {showChatLog ? (
        <IconButton
          iconName="24/CommentOutline"
          label="会話ログ"
          isProcessing={false}
          onClick={handleToggleChatLog}
        />
      ) : (
        <IconButton
          iconName="24/CommentFill"
          label="会話ログ"
          isProcessing={false}
          disabled={chatLog.length <= 0}
          onClick={handleToggleChatLog}
        />
      )}
    </div>
  );

  return (
    <>
      <div className="absolute z-10 m-24">{renderButtons()}</div>
      {showChatLog && <ChatLog messages={chatLog} />}
      {showSettings && (
        <Settings
          openAiKey={openAiKey}
          systemPrompt={systemPrompt}
          chatLog={chatLog}
          koeiroParam={koeiroParam}
          onChangeAiKey={handleAiKeyChange}
          onClickClose={handleToggleSettings}
          onChangeSystemPrompt={handleChangeSystemPrompt}
          onChangeChatLog={onChangeChatLog}
          onChangeKoeiroParam={handleChangeKoeiroParam}
          onClickOpenVrmFile={handleClickOpenVrmFile}
        />
      )}
      {!showChatLog && assistantMessage && <AssistantText message={assistantMessage} />}
      <input
        type="file"
        className="hidden"
        accept=".vrm"
        ref={fileInputRef}
        onChange={handleChangeVrmFile}
      />
    </>
  );
};

export default Menu; // Export the component as default

