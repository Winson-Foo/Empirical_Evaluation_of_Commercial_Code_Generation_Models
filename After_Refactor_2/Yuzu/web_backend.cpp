#include <array>
#include <mutex>
#include <string>

#include <httplib.h>

#include "common/logging/log.h"

namespace WebService {

constexpr std::array<const char, 1> API_VERSION{'1'};

constexpr std::size_t TIMEOUT_SECONDS = 30;

struct Client::Impl {
    Impl(const std::string& host, const std::string& username, const std::string& token)
        : host_(host), username_(username), token_(token) {
        std::scoped_lock lock{jw_cache_.mutex};
        if (username == jw_cache_.username && token == jw_cache_.token) {
            jwt_ = jw_cache_.jwt;
        }
    }

    WebResult GenericRequest(const std::string& method, const std::string& path,
                             const std::string& data, bool allow_anonymous,
                             const std::string& accept) const {
        if (jwt_.empty()) {
            UpdateJWT();
        }

        if (jwt_.empty() && !allow_anonymous) {
            LOG_ERROR(WebService, "Credentials must be provided for authenticated requests");
            return WebResult{WebResult::Code::CredentialsMissing, "Credentials needed", ""};
        }

        auto result = GenericRequest(method, path, data, accept, jwt_);
        if (result.result_string == "401") {
            // Try again with new JWT
            UpdateJWT();
            result = GenericRequest(method, path, data, accept, jwt_);
        }

        return result;
    }

    WebResult GenericRequest(const std::string& method, const std::string& path,
                             const std::string& data, const std::string& accept,
                             const std::string& jwt = "", const std::string& username = "",
                             const std::string& token = "") const {
        httplib::Client cli{host_};
        if (!cli.is_valid()) {
            LOG_ERROR(WebService, "Client is invalid, skipping request!");
            return {};
        }

        cli.set_connection_timeout(TIMEOUT_SECONDS);
        cli.set_read_timeout(TIMEOUT_SECONDS);
        cli.set_write_timeout(TIMEOUT_SECONDS);

        httplib::Headers params;
        if (!jwt.empty()) {
            params = {
                {std::string("Authorization"), "Bearer " + jwt},
            };
        } else if (!username.empty()) {
            params = {
                {"x-username", username},
                {"x-token", token},
            };
        }

        params.emplace("api-version", std::string(API_VERSION.begin(), API_VERSION.end()));
        if (method != "GET") {
            params.emplace("Content-Type", "application/json");
        }

        httplib::Request request;
        request.method = method;
        request.path = path;
        request.headers = params;
        request.body = data;

        httplib::Response response;
        httplib::Error error;
        if (!cli.send(request, response, error)) {
            LOG_ERROR(WebService, "{} to {} returned null (httplib Error: {})", method, host_ + path,
                      httplib::to_string(error));
            return WebResult{WebResult::Code::LibError, "Null response", ""};
        }

        if (response.status >= 400) {
            LOG_ERROR(WebService, "{} to {} returned error status code: {}", method, host_ + path,
                      response.status);
            return WebResult{WebResult::Code::HttpError, std::to_string(response.status), ""};
        }

        auto content_type = response.headers.find("content-type");

        if (content_type == response.headers.end()) {
            LOG_ERROR(WebService, "{} to {} returned no content", method, host_ + path);
            return WebResult{WebResult::Code::WrongContent, "", ""};
        }

        if (content_type->second.find(accept) == std::string::npos) {
            LOG_ERROR(WebService, "{} to {} returned wrong content: {}", method, host_ + path,
                      content_type->second);
            return WebResult{WebResult::Code::WrongContent, "Wrong content", ""};
        }
        return WebResult{WebResult::Code::Success, "", response.body};
    }

    // Retrieve a new JWT from given username and token
    void UpdateJWT() const {
        if (username_.empty() || token_.empty()) {
            return;
        }

        auto result = GenericRequest("POST", "/jwt/internal", "", "text/html", "", username_, token_);
        if (result.result_code != WebResult::Code::Success) {
            LOG_ERROR(WebService, "UpdateJWT failed");
        } else {
            std::scoped_lock lock{jw_cache_.mutex};
            jw_cache_.username = username_;
            jw_cache_.token = token_;
            jw_cache_.jwt = jwt_ = result.returned_data;
        }
    }

    std::string host_;
    std::string username_;
    std::string token_;
    mutable std::string jwt_;
    mutable httplib::Client cli_;

    struct JWTCache {
        std::mutex mutex;
        std::string username;
        std::string token;
        std::string jwt;
    };
    static inline JWTCache jw_cache_;
};

Client::Client(const std::string& host, const std::string& username, const std::string& token)
    : impl{std::make_unique<Impl>(host, username, token)} {}

Client::~Client() = default;

WebResult Client::PostJson(const std::string& path, const std::string& data, bool allow_anonymous) const {
    return impl->GenericRequest("POST", path, data, allow_anonymous, "application/json");
}

WebResult Client::GetJson(const std::string& path, bool allow_anonymous) const {
    return impl->GenericRequest("GET", path, "", allow_anonymous, "application/json");
}

WebResult Client::DeleteJson(const std::string& path, const std::string& data,
                             bool allow_anonymous) const {
    return impl->GenericRequest("DELETE", path, data, allow_anonymous, "application/json");
}

WebResult Client::GetPlain(const std::string& path, bool allow_anonymous) const {
    return impl->GenericRequest("GET", path, "", allow_anonymous, "text/plain");
}

WebResult Client::GetImage(const std::string& path, bool allow_anonymous) const {
    return impl->GenericRequest("GET", path, "", allow_anonymous, "image/png");
}

WebResult Client::GetExternalJWT(const std::string& audience) const {
    return impl->GenericRequest("POST", "/jwt/external/" + audience, "", false, "text/html");
}

}  // namespace WebService