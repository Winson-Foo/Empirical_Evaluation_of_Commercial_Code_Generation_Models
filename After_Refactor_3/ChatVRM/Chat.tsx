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