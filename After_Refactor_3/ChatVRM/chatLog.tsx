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