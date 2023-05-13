// menu.tsx
import React, { useCallback, useContext, useRef, useState } from "react";
import { IconButton } from "./iconButton";
import { ChatLog } from "./chatLog";
import { Settings } from "./settings";
import { AssistantText } from "./assistantText";
import { ViewerContext } from "@/features/vrmViewer/viewerContext";
import { KoeiroParam } from "@/features/constants/koeiroParam";
import { Message } from "@/features/messages/messages";

interface MenuProps {
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

export const Menu: React.FC<MenuProps> = ({
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

  const handleSystemPromptChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      onChangeSystemPrompt(event.target.value);
    },
    [onChangeSystemPrompt]
  );

  const handleAiKeyChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onChangeAiKey(event.target.value);
    },
    [onChangeAiKey]
  );

  const handleKoeiroParamChange = useCallback(
    (x: number, y: number) => {
      onChangeKoeiromapParam({
        speakerX: x,
        speakerY: y,
      });
    },
    [onChangeKoeiromapParam]
  );

  const handleOpenVrmFileClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleVrmFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
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
    },
    [viewer]
  );

  return (
    <>
      <div className="absolute z-10 m-24">
        <div className="grid grid-flow-col gap-[8px]">
          <IconButton
            iconName="24/Menu"
            label="Settings"
            isProcessing={false}
            onClick={() => setShowSettings(true)}
          ></IconButton>
          {showChatLog ? (
            <IconButton
              iconName="24/CommentOutline"
              label="Hide Chat Log"
              isProcessing={false}
              onClick={() => setShowChatLog(false)}
            />
          ) : (
            <IconButton
              iconName="24/CommentFill"
              label="Show Chat Log"
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
          onClose={() => setShowSettings(false)}
          onAiKeyChange={handleAiKeyChange}
          onSystemPromptChange={handleSystemPromptChange}
          onChatLogChange={onChangeChatLog}
          onKoeiroParamChange={handleKoeiroParamChange}
          onOpenVrmFileClick={handleOpenVrmFileClick}
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
        onChange={handleVrmFileChange}
      />
    </>
  );
};

// settings.tsx
import React from "react";
import { ChatLog } from "./chatLog";
import { KoeiroParam } from "@/features/constants/koeiroParam";

interface SettingsProps {
  openAiKey: string;
  chatLog: Message[];
  systemPrompt: string;
  koeiroParam: KoeiroParam;
  onClose: () => void;
  onAiKeyChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onSystemPromptChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onChatLogChange: (index: number, text: string) => void;
  onKoeiroParamChange: (x: number, y: number) => void;
  onOpenVrmFileClick: () => void;
}

export const Settings: React.FC<SettingsProps> = ({
  openAiKey,
  chatLog,
  systemPrompt,
  koeiroParam,
  onClose,
  onAiKeyChange,
  onSystemPromptChange,
  onChatLogChange,
  onKoeiroParamChange,
  onOpenVrmFileClick,
}) => {
  return (
    <div className="fixed top-0 left-0 w-screen h-screen bg-gray-200 flex justify-center items-center">
      <div className="bg-white max-w-[800px] mx-auto p-8 space-y-8">
        <h2 className="text-lg font-bold text-gray-800">Settings</h2>
        <div className="space-y-4">
          <div>
            <label className="block font-medium text-gray-800">
              AI Key
            </label>
            <input
              type="text"
              className="w-full border border-gray-300 rounded-md p-2"
              value={openAiKey}
              onChange={onAiKeyChange}
            />
          </div>
          <div>
            <label className="block font-medium text-gray-800">
              System Prompt
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-md p-2"
              rows={2}
              value={systemPrompt}
              onChange={onSystemPromptChange}
            />
          </div>
          <div>
            <label className="block font-medium text-gray-800">
              Koeiromap Speaker Position (X, Y)
            </label>
            <div className="flex space-x-2 items-center">
              <input
                type="number"
                className="w-1/4 border border-gray-300 rounded-md p-2"
                value={koeiroParam.speakerX}
                onChange={(event) =>
                  onKoeiroParamChange(
                    Number(event.target.value),
                    koeiroParam.speakerY
                  )
                }
              />
              <span className="text-gray-800 font-medium">,</span>
              <input
                type="number"
                className="w-1/4 border border-gray-300 rounded-md p-2"
                value={koeiroParam.speakerY}
                onChange={(event) =>
                  onKoeiroParamChange(
                    koeiroParam.speakerX,
                    Number(event.target.value)
                  )
                }
              />
            </div>
          </div>
          <div>
            <label className="block font-medium text-gray-800">
              Chat Log
            </label>
            <ChatLog messages={chatLog} isEditable onChange={onChatLogChange} />
          </div>
          <div>
            <label className="block font-medium text-gray-800">
              VRM File
            </label>
            <div className="flex space-x-2 items-center">
              <button
                className="px-4 py-2 bg-blue-500 text-white rounded-md"
                onClick={onOpenVrmFileClick}
              >
                Select File
              </button>
            </div>
          </div>
        </div>
        <button
          className="bg-red-500 text-white px-4 py-2 rounded-md"
          onClick={onClose}
        >
          Close
        </button>
      </div>
    </div>
  );
};

// chatLog.tsx
import React from "react";
import { Message } from "@/features/messages/messages";

interface ChatLogProps {
  messages: Message[];
  isEditable?: boolean;
  onChange?: (index: number, text: string) => void;
}

export const ChatLog: React.FC<ChatLogProps> = ({
  messages,
  isEditable = false,
  onChange,
}) => {
  return (
    <div className="border border-gray-300 rounded-md overflow-y-auto max-h-[300px] p-2">
      {messages.map((message, index) => (
        <div key={index} className="flex space-x-2">
          <span className="text-gray-500">{message.timestamp}</span>
          <div className="flex-1">
            {isEditable ? (
              <input
                type="text"
                className="w-full border border-gray-300 rounded-md p-2"
                value={message.text}
                onChange={(event) => onChange?.(index, event.target.value)}
              />
            ) : (
              <span>{message.text}</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

// assistantText.tsx
import React from "react";

interface AssistantTextProps {
  message: string;
}

export const AssistantText: React.FC<AssistantTextProps> = ({ message }) => {
  return (
    <div className="fixed bottom-0 left-0 w-screen p-4 bg-gray-800 text-white">
      <div className="max-w-lg mx-auto">{message}</div>
    </div>
  );
};

// iconButton.tsx
import React from "react";

interface IconButtonProps {
  iconName: string;
  label: string;
  isProcessing: boolean;
  disabled?: boolean;
  onClick: () => void;
}

export const IconButton: React.FC<IconButtonProps> = ({
  iconName,
  label,
  isProcessing,
  disabled = false,
  onClick,
}) => {
  return (
    <button
      className="flex space-x-2 items-center bg-white text-gray-800 rounded-md shadow p-2 disabled:opacity-50"
      disabled={disabled}
      onClick={onClick}
    >
      <img
        src={`/icons/${iconName}.svg`}
        alt=""
        className={`w-6 h-6 ${isProcessing ? "animate-spin" : ""}`}
      />
      <span className="font-medium">{label}</span>
    </button>
  );
}

