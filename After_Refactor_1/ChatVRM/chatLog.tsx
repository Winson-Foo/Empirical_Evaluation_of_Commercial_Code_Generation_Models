import { useEffect, useRef } from "react";
import { Message } from "@/features/messages/messages";

type Props = {
  messages: Message[];
};

type ScrollOptions = {
  behavior: "auto" | "smooth";
  block?: "start" | "center" | "end" | "nearest";
  inline?: "start" | "center" | "end" | "nearest";
};

const useScrollIntoView = (options: ScrollOptions = {}) => {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    ref.current?.scrollIntoView(options);
  }, [options]);

  return ref;
};

export const ChatLog = ({ messages }: Props) => {
  const chatScrollRef = useScrollIntoView({
    behavior: "auto",
    block: "center",
  });

  useScrollIntoView({
    ref: chatScrollRef,
    behavior: "smooth",
    block: "center",
    deps: [messages],
  });

  return (
    <div className="absolute w-col-span-6 max-w-full h-[100svh] pb-64">
      <div className="max-h-full px-16 pt-104 pb-64 overflow-y-auto scroll-hidden">
        {messages.map((msg, i) => (
          <div key={i} ref={messages.length - 1 === i ? chatScrollRef : null}>
            <Chat role={msg.role} message={msg.content} />
          </div>
        ))}
      </div>
    </div>
  );
};

const Chat = ({ role, message }: { role: string; message: string }) => {
  const isAssistant = role === "assistant";
  const roleColor = isAssistant ? "bg-secondary text-white " : "bg-base text-primary";
  const roleText = isAssistant ? "text-secondary" : "text-primary";
  const offsetX = isAssistant ? "pr-40" : "pl-40";
  const borderRadiusTop = isAssistant ? "rounded-tr-8" : "rounded-tl-8";
  const borderRadiusBottom = isAssistant ? "rounded-bl-8" : "rounded-br-8";
  const containerStyles = `mx-auto max-w-sm my-16 ${offsetX}`;
  const roleStyles = `px-24 py-8 font-Montserrat font-bold tracking-wider ${roleColor} ${borderRadiusTop}`;
  const messageStyles = `px-24 py-16 bg-white ${roleText} ${borderRadiusBottom}`;

  return (
    <div className={containerStyles}>
      <div className={roleStyles}>{isAssistant ? "CHARACTER" : "YOU"}</div>
      <div className={messageStyles}>
        <div className="typography-16 font-M_PLUS_2 font-bold">{message}</div>
      </div>
    </div>
  );
};