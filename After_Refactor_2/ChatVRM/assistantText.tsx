const MESSAGE = 'message';
const CHARACTER_NAME = 'CHARACTER';

const AssistantText = ({ [MESSAGE]: message }: { [MESSAGE]: string }) => {
  return (
    <div className="assistant-text">
      <div className="container">
        <div className="message-container">
          <div className="character-name">{CHARACTER_NAME}</div>
          <div className="message">{getMessageWithoutTags(message)}</div>
        </div>
      </div>
    </div>
  );
};

function getMessageWithoutTags(message: string): string {
  return message.replace(/\[([a-zA-Z]*?)\]/g, '');
}