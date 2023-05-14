namespace WebService {

enum class ResultCode {
    Success,
    HttpError,
    LibError,
    CredentialsMissing,
    WrongContent
};

struct WebResult {
    ResultCode result_code;
    std::string message;
    std::string returned_data;
};

class Client::Impl {
public:
    Impl(const std::string& host, const std::string& username, const std::string& token)
        : host_(host), username_(username), token_(token) {
        std::scoped_lock lock(jwt_cache_.mutex);
        if (username_ == jwt_cache_.username && token_ == jwt_cache_.token) {
            jwt_ = jwt_cache_.jwt;
        }
    }

    WebResult PostJson(const std::string& path, const std::string& data, bool allow_anonymous) {
        return GenericRequest("POST", path, data, allow_anonymous, "application/json");
    }

    WebResult GetJson(const std::string& path, bool allow_anonymous) {
        return GenericRequest("GET", path, "", allow_anonymous, "application/json");
    }

    WebResult DeleteJson(const std::string& path, const std::string& data, bool allow_anonymous) {
        return GenericRequest("DELETE", path, data, allow_anonymous, "application/json");
    }

    WebResult GetPlain(const std::string& path, bool allow_anonymous) {
        return GenericRequest("GET", path, "", allow_anonymous, "text/plain");
    }

    WebResult GetImage(const std::string& path, bool allow_anonymous) {
        return GenericRequest("GET", path, "", allow_anonymous, "image/png");
    }

    WebResult GetExternalJWT(const std::string& audience) {
        return GenericRequest("POST", fmt::format("/jwt/external/{}", audience), "", false, "text/html");
    }

private:
    WebResult GenericRequest(const std::string& method, const std::string& path, const std::string& requestData,
                             bool allowAnonymous, const std::string& accept) {
        if (jwt_.empty()) {
            UpdateJWT();
        }

        if (jwt_.empty() && !allowAnonymous) {
            LOG_ERROR(WebService, "Credentials must be provided for authenticated requests");
            return {ResultCode::CredentialsMissing, "Credentials needed", ""};
        }

        auto result = GenericRequest(method, path, requestData, accept, jwt_);
        if (result.result_code == ResultCode::HttpError && result.returned_data == "401") {
            // Try again with new JWT
            UpdateJWT();
            result = GenericRequest(method, path, requestData, accept, jwt_);
        }

        return result;
    }

    WebResult GenericRequest(const std::string& method, const std::string& path, const std::string& requestData,
                             const std::string& accept, const std::string& jwt = "", const std::string& username = "",
                             const std::string& token = "") {
        if (!httpClient_) {
            httpClient_ = std::make_unique<httplib::Client>(host_);
        }

        if (!httpClient_->is_valid()) {
            LOG_ERROR(WebService, "Client is invalid, skipping request!");
            return {};
        }

        httpClient_->set_connection_timeout(TIMEOUT_SECONDS);
        httpClient_->set_read_timeout(TIMEOUT_SECONDS);
        httpClient_->set_write_timeout(TIMEOUT_SECONDS);

        httplib::Headers headers;
        if (!jwt.empty()) {
            headers = {{"Authorization", fmt::format("Bearer {}", jwt)},};
        } else if (!username.empty() && !token.empty()) {
            headers = {{"x-username", username}, {"x-token", token},};
        }

        headers.emplace("api-version", std::string(API_VERSION.begin(), API_VERSION.end()));
        if (method != "GET") {
            headers.emplace("Content-Type", "application/json");
        }

        httplib::Request request;
        request.method = method;
        request.path = path;
        request.headers = headers;
        request.body = requestData;

        httplib::Response response;
        httplib::Error error;

        if (!httpClient_->send(request, response, error)) {
            LOG_ERROR(WebService, "{} to {} returned null (httplib Error: {})", method, host_ + path,
                      httplib::to_string(error));
            return {ResultCode::LibError, "Null response", ""};
        }

        if (response.status >= 400) {
            LOG_ERROR(WebService, "{} to {} returned error status code: {}", method, host_ + path,
                      response.status);
            return {ResultCode::HttpError, std::to_string(response.status), ""};
        }

        auto contentTypeHeader = response.headers.find("content-type");

        if (contentTypeHeader == response.headers.end()) {
            LOG_ERROR(WebService, "{} to {} returned no content", method, host_ + path);
            return {ResultCode::WrongContent, "", ""};
        }

        if (contentTypeHeader->second.find(accept) == std::string::npos) {
            LOG_ERROR(WebService, "{} to {} returned wrong content: {}", method, host_ + path,
                      contentTypeHeader->second);
            return {ResultCode::WrongContent, "Wrong content", ""};
        }

        return {ResultCode::Success, "", response.body};
    }

    void UpdateJWT() {
        if (username_.empty() || token_.empty()) {
            return;
        }

        auto result = GenericRequest("POST", "/jwt/internal", "", "text/html", "", username_, token_);
        if (result.result_code != ResultCode::Success) {
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
    std::unique_ptr<httplib::Client> httpClient_;

    struct JWTCache {
        std::mutex mutex;
        std::string username;
        std::string token;
        std::string jwt;
    };
    static JWTCache& GetJWTCache() {
        static JWTCache jwt_cache;
        return jwt_cache;
    }
    static constexpr std::array<const char, 1> API_VERSION{'1'};
    static constexpr std::size_t TIMEOUT_SECONDS = 30;
};

Client::Client(const std::string& host, const std::string& username, const std::string& token)
    : impl_(std::make_unique<Impl>(host, username, token)) {}

Client::~Client() = default;

WebResult Client::PostJson(const std::string& path, const std::string& data, bool allowAnonymous) {
    return impl_->PostJson(path, data, allowAnonymous);
}

WebResult Client::GetJson(const std::string& path, bool allowAnonymous) {
    return impl_->GetJson(path, allowAnonymous);
}

WebResult Client::DeleteJson(const std::string& path, const std::string& data, bool allowAnonymous) {
    return impl_->DeleteJson(path, data, allowAnonymous);
}

WebResult Client::GetPlain(const std::string& path, bool allowAnonymous) {
    return impl_->GetPlain(path, allowAnonymous);
}

WebResult Client::GetImage(const std::string& path, bool allowAnonymous) {
    return impl_->GetImage(path, allowAnonymous);
}

WebResult Client::GetExternalJWT(const std::string& audience) {
    return impl_->GetExternalJWT(audience);
}

} // namespace WebService