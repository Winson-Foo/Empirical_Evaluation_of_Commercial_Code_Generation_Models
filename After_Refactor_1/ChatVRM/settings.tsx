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