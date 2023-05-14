#include <chrono>
#include <future>
#include <vector>
#include "announce_multiplayer_session.h"
#include "common/announce_multiplayer_room.h"
#include "common/assert.h"
#include "common/settings.h"
#include "network/network.h"
#include "web_service/announce_room_json.h"

namespace Core {

// Time between room is announced to web_service
static constexpr std::chrono::seconds announce_time_interval(15);

AnnounceMultiplayerSession::AnnounceMultiplayerSession(Network::RoomNetwork& room_network_)
  : room_network_{room_network_},
    backend_{std::make_unique<WebService::RoomJson>(
      Settings::values.web_api_url.GetValue(),
      Settings::values.yuzu_username.GetValue(),
      Settings::values.yuzu_token.GetValue()
    )} {}

WebService::WebResult AnnounceMultiplayerSession::Register() const {
  const auto room = room_network_.GetRoom().lock();
  if (!room) {
    return {
      WebService::WebResult::Code::LibError,
      "Network is not initialized",
      ""
    };
  }
  if (room->GetState() != Network::Room::State::Open) {
    return {
      WebService::WebResult::Code::LibError,
      "Room is not open",
      ""
    };
  }
  UpdateBackendData(*room);
  const auto result = backend_->Register();
  if (result.result_code != WebService::WebResult::Code::Success) {
    return result;
  }
  LOG_INFO(WebService, "Room has been registered");
  room->SetVerifyUID(result.returned_data);
  registered_ = true;
  return {WebService::WebResult::Code::Success, "", ""};
}

void AnnounceMultiplayerSession::Start() {
  if (announce_multiplayer_thread_) {
    Stop();
  }
  shutdown_event_.Reset();
  announce_multiplayer_thread_ =
    std::make_unique<std::thread>(&AnnounceMultiplayerSession::AnnounceMultiplayerLoop, this);
}

void AnnounceMultiplayerSession::Stop() {
  if (announce_multiplayer_thread_) {
    shutdown_event_.Set();
    announce_multiplayer_thread_->join();
    announce_multiplayer_thread_.reset();
    backend_->Delete();
    registered_ = false;
  }
}

AnnounceMultiplayerSession::CallbackHandle AnnounceMultiplayerSession::BindErrorCallback(
  const std::function<void(const WebService::WebResult&)>& function) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  const auto handle = std::make_shared<std::function<void(const WebService::WebResult&)>>(function);
  error_callbacks_.insert(handle);
  return handle;
}

void AnnounceMultiplayerSession::UnbindErrorCallback(CallbackHandle handle) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  error_callbacks_.erase(handle);
}

AnnounceMultiplayerSession::~AnnounceMultiplayerSession() {
  Stop();
}

AnnounceMultiplayerRoom::RoomList AnnounceMultiplayerSession::GetRoomList() const {
  return backend_->GetRoomList();
}

bool AnnounceMultiplayerSession::IsRunning() const {
  return announce_multiplayer_thread_ != nullptr;
}

void AnnounceMultiplayerSession::UpdateCredentials() {
  ASSERT_MSG(!IsRunning(), "Credentials can only be updated when session is not running");
  backend_ = std::make_unique<WebService::RoomJson>(
    Settings::values.web_api_url.GetValue(),
    Settings::values.yuzu_username.GetValue(),
    Settings::values.yuzu_token.GetValue()
  );
}

void AnnounceMultiplayerSession::UpdateBackendData(const Network::Room& room) {
  const auto room_information = room.GetRoomInformation();
  const auto memberlist = room.GetRoomMemberList();
  backend_->SetRoomInformation(
    room_information.name,
    room_information.description,
    room_information.port,
    room_information.member_slots,
    Network::network_version,
    room.HasPassword(),
    room_information.preferred_game
  );
  backend_->ClearPlayers();
  for (const auto& member : memberlist) {
    backend_->AddPlayer(member);
  }
}

void AnnounceMultiplayerSession::AnnounceMultiplayerLoop() {
  const auto ErrorCallback = [this] (const WebService::WebResult& result) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    for (const auto& callback : error_callbacks_) {
      (*callback)(result);
    }
  };

  if (!registered_) {
    const auto result = Register();
    if (result.result_code != WebService::WebResult::Code::Success) {
      ErrorCallback(result);
      return;
    }
  }

  auto update_time = std::chrono::steady_clock::now();
  std::future<WebService::WebResult> future;
  while (!shutdown_event_.WaitUntil(update_time)) {
    update_time += announce_time_interval;
    const auto room = room_network_.GetRoom().lock();
    if (!room) {
      break;
    }
    if (room->GetState() != Network::Room::State::Open) {
      break;
    }
    UpdateBackendData(*room);
    const auto result = backend_->Update();
    if (result.result_code != WebService::WebResult::Code::Success) {
      ErrorCallback(result);
    }
    if (result.result_string == "404") {
      registered_ = false;
      const auto register_result = Register();
      if (register_result.result_code != WebService::WebResult::Code::Success) {
        ErrorCallback(register_result);
      }
    }
  }
}

} // namespace Core