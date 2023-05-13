// ChatLog.tsx
import { useEffect, useRef } from "react";
import { Message } from "@/features/messages/messages";
import { Chat } from "./Chat";

type Props = {
  messages: Message[];
};

export const ChatLog = ({ messages }: Props) => {
  const chatScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatScrollRef.current?.scrollIntoView({
      behavior: "auto",
      block: "center",
    });
  }, []);

  useEffect(() => {
    chatScrollRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [messages]);

  return (
    <div className="absolute w-col-span-6 max-w-full h-[100svh] pb-64">
      <div className="max-h-full px-16 pt-104 pb-64 overflow-y-auto scroll-hidden">
        {messages.map((message, index) => {
          const isLastMessage = messages.length - 1 === index;
          const messageRef = isLastMessage ? chatScrollRef : null;

          return (
            <div key={index} ref={messageRef}>
              <Chat role={message.role} content={message.content} />
            </div>
          );
        })}
      </div>
    </div>
  );
};


// Chat.tsx
import React from "react";

type Props = {
  role: string;
  content: string;
};

export const Chat = ({ role, content }: Props) => {
  const isAssistant = role === "assistant";
  const roleColor = isAssistant ? "bg-secondary text-white " : "bg-base text-primary";
  const roleText = isAssistant ? "text-secondary" : "text-primary";
  const offsetX = role === "user" ? "pl-40" : "pr-40";
  
  return (
    <div className={`mx-auto max-w-sm my-16 ${offsetX}`}>
      <div className={`px-24 py-8 rounded-t-8 font-Montserrat font-bold tracking-wider ${roleColor}`}>
        {isAssistant ? "CHARACTER" : "YOU"}
      </div>
      <div className="px-24 py-16 bg-white rounded-b-8">
        <div className={`typography-16 font-M_PLUS_2 font-bold ${roleText}`}>
          {content}
        </div>
      </div>
    </div>
  );
};

