// SPDX-License-Identifier: GPL-2.0-or-later

#include "AnnounceMultiplayerSession.h"

namespace Core {

typedef std::chrono::seconds TimeDuration;
typedef std::shared_ptr<Network::Room> RoomPtr;
typedef std::function<void(const WebService::WebResult&)> ErrorCallbackFn;
typedef std::shared_ptr<ErrorCallbackFn> ErrorCallbackPtr;
typedef std::set<ErrorCallbackPtr> ErrorCallbackSet;

class AnnounceMultiplayerLoop {
public:
    AnnounceMultiplayerLoop(AnnounceBackend& backend, RoomNetwork& room_network,
                            const TimeDuration& announce_time_interval,
                            const ErrorCallbackSet& error_callbacks)
        : backend_(backend), room_network_(room_network),
          announce_time_interval_(announce_time_interval), error_callbacks_(error_callbacks),
          shutdown_event_(false) {}

    void operator()() {
        // Invokes all current bound error callbacks.
        const auto ErrorCallback = [this](const WebService::WebResult& result) {
            for (auto callback : error_callbacks_) {
                (*callback)(result);
            }
        };

        bool is_registered = false;
        RoomPtr room;

        auto update_time = std::chrono::steady_clock::now();

        while (!shutdown_event_.WaitUntil(update_time)) {
            update_time += announce_time_interval_;

            room = room_network_.GetRoom().lock();
            if (!room) {
                break;
            }

            if (room->GetState() != Network::Room::State::Open) {
                break;
            }

            if (!is_registered) {
                WebService::WebResult result = backend_.Register();
                if (result.result_code != WebService::WebResult::Code::Success) {
                    ErrorCallback(result);
                    break;
                }

                room->SetVerifyUID(result.returned_data);
                LOG_INFO(WebService, "Room has been registered");

                is_registered = true;
            }

            backend_.UpdateRoomData(room);
            WebService::WebResult result = backend_.Update();
            if (result.result_code != WebService::WebResult::Code::Success) {
                ErrorCallback(result);
            }

            if (result.result_string == "404") {
                is_registered = false;
            }
        }

        if (is_registered) {
            backend_.Delete();
            room->SetVerifyUID("");
        }
    }

private:
    AnnounceBackend& backend_;
    RoomNetwork& room_network_;
    TimeDuration announce_time_interval_;
    ErrorCallbackSet error_callbacks_;
    ShutdownEvent shutdown_event_;
};

AnnounceMultiplayerSession::AnnounceMultiplayerSession(AnnounceBackend& backend, RoomNetwork& room_network)
    : backend_(backend), room_network_(room_network), is_running_(false) {}

AnnounceMultiplayerSession::~AnnounceMultiplayerSession() {
    Stop();
}

void AnnounceMultiplayerSession::Start() {
    Stop();

    error_callbacks_.clear();
    error_callbacks_.insert(std::make_shared<ErrorCallbackFn>());

    AnnounceMultiplayerLoop loop(backend_, room_network_, announce_time_interval_, error_callbacks_);

    is_running_ = true;
    thread_ = std::make_unique<std::thread>(loop);
}

void AnnounceMultiplayerSession::Stop() {
    if (is_running_) {
        is_running_ = false;
        thread_->join();
        thread_.reset();
    }
}

void AnnounceMultiplayerSession::BindErrorCallback(ErrorCallbackFn function) {
    error_callbacks_.insert(std::make_shared<ErrorCallbackFn>(function));
}

void AnnounceMultiplayerSession::UnbindErrorCallback(const ErrorCallbackFn& function) {
    for (auto it = error_callbacks_.begin(); it != error_callbacks_.end(); ++it) {
        if (*(*it) == function) {
            error_callbacks_.erase(it);
            break;
        }
    }
}

RoomList AnnounceMultiplayerSession::GetRoomList() {
    return backend_.GetRoomList();
}

} // namespace Core

