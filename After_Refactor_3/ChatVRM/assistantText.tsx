// ChatBubble component
const ChatBubble = ({ characterName, message }: { characterName: string, message: string }) => {
  return (
    <div className="bg-white rounded-8">
      <div className="px-24 py-8 bg-secondary rounded-t-8 text-white font-Montserrat font-bold tracking-wider">
        {characterName.toUpperCase()}
      </div>
      <div className="px-24 py-16">
        <div className="line-clamp-4 text-secondary typography-16 font-M_PLUS_2 font-bold">
          {message.replace(/\[([a-zA-Z]*?)\]/g, "") /* remove brackets */}
        </div>
      </div>
    </div>
  );
};

// AssistantText component
export const AssistantText = ({ message }: { message: string }) => {
  return (
    <div className="absolute bottom-0 left-0 mb-104  w-full">
      <div className="mx-auto max-w-4xl w-full p-16">
        <ChatBubble characterName="CHARACTER" message={message} />
      </div>
    </div>
  );
}; 