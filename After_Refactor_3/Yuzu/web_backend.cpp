// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <mutex>
#include <string>
#include <memory>

#include <fmt/format.h>
#include <httplib.h>

#include "common/logging/log.h"
#include "web_service/web_backend.h"
#include "web_service/web_result.h"

namespace {
    constexpr std::size_t TIMEOUT_SECONDS = 30;
    constexpr std::array<const char, 1> API_VERSION{'1'};
    struct JWTCache {
        std::mutex mutex;
        std::string username;
        std::string token;
        std::string jwt;
    };
}

namespace WebService {
    // Private implementation of the web service client
    class Client::Impl {
    public:
        Impl(std::string host, std::string username, std::string token) :
            host_(std::move(host)),
            username_(std::move(username)),
            token_(std::move(token)) {
            std::scoped_lock lock(jwt_cache_.mutex);
            if (username_ == jwt_cache_.username && token_ == jwt_cache_.token) {
                jwt_ = jwt_cache_.jwt;
            }
        }

        WebResult PostJson(const std::string& path, const std::string& data, bool allow_anonymous) {
            return GenericRequest("POST", path, data, "application/json", allow_anonymous);
        }

        WebResult GetJson(const std::string& path, bool allow_anonymous) {
            return GenericRequest("GET", path, "", "application/json", allow_anonymous);
        }

        WebResult DeleteJson(const std::string& path, const std::string& data, bool allow_anonymous) {
            return GenericRequest("DELETE", path, data, "application/json", allow_anonymous);
        }

        WebResult GetPlain(const std::string& path, bool allow_anonymous) {
            return GenericRequest("GET", path, "", "text/plain", allow_anonymous);
        }

        WebResult GetImage(const std::string& path, bool allow_anonymous) {
            return GenericRequest("GET", path, "", "image/png", allow_anonymous);
        }

    private:
        // Generic function for making web requests 
        WebResult GenericRequest(const std::string& method, const std::string& path,
                                 const std::string& data, const std::string& accept, bool allow_anonymous=false) {
            if (!allow_anonymous && jwt_.empty()) {
                LOG_ERROR(WebService, "Credentials must be provided for authenticated requests");
                return {WebResult::Code::CredentialsMissing, "Credentials needed", ""};
            }

            if (jwt_.empty()) {
                UpdateJWT();
            }

            httplib::Headers headers;
            headers.emplace("api-version", std::string(API_VERSION.begin(), API_VERSION.end()));

            if (!jwt_.empty()) {
                headers.emplace("Authorization", fmt::format("Bearer {}", jwt_));
            } else {
                headers.emplace("x-username", username_);
                headers.emplace("x-token", token_);
            }

            if (method != "GET") {
                headers.emplace("Content-Type", "application/json");
            }

            httplib::Request request;
            request.method = method;
            request.path = path;
            request.headers = headers;
            request.body = data;

            if (!cli_) {
                cli_ = std::make_unique<httplib::Client>(host_);
            }

            cli_->set_connection_timeout(TIMEOUT_SECONDS);
            cli_->set_read_timeout(TIMEOUT_SECONDS);
            cli_->set_write_timeout(TIMEOUT_SECONDS);

            httplib::Response response;
            httplib::Error error;

            if (!cli_->send(request, response, error)) {
                LOG_ERROR(WebService, "{} to {} returned null (httplib Error: {})", method, host_ + path,
                          httplib::to_string(error));
                return {WebResult::Code::LibError, "Null response", ""};
            }

            if (response.status >= 400) {
                LOG_ERROR(WebService, "{} to {} returned error status code: {}", method, host_ + path,
                          response.status);
                return {WebResult::Code::HttpError, std::to_string(response.status), ""};
            }

            auto content_type = response.headers.find("content-type");

            if (content_type == response.headers.end()) {
                LOG_ERROR(WebService, "{} to {} returned no content", method, host_ + path);
                return {WebResult::Code::WrongContent, "", ""};
            }

            if (content_type->second.find(accept) == std::string::npos) {
                LOG_ERROR(WebService, "{} to {} returned wrong content: {}", method, host_ + path,
                          content_type->second);
                return {WebResult::Code::WrongContent, "Wrong content", ""};
            }

            return {WebResult::Code::Success, "", response.body};
        }

        // Retrieve a new JWT from given username and token
        void UpdateJWT() {
            if (username_.empty() || token_.empty()) {
                return;
            }

            auto result = GenericRequest("POST", "/jwt/internal", "", "text/html", true);
            if (result.result_code != WebResult::Code::Success) {
                LOG_ERROR(WebService, "UpdateJWT failed");
            } else {
                std::scoped_lock lock(jwt_cache_.mutex);
                jwt_cache_.username = username_;
                jwt_cache_.token = token_;
                jwt_cache_.jwt = jwt_ = result.returned_data;
            }
        }

        std::string host_;
        std::string username_;
        std::string token_;
        std::string jwt_;
        std::unique_ptr<httplib::Client> cli_;
        static inline JWTCache jwt_cache_;
    };

    Client::Client(std::string host, std::string username, std::string token)
        : impl_{std::make_unique<Impl>(std::move(host), std::move(username), std::move(token))} {}

    Client::~Client() = default;

} // namespace WebService