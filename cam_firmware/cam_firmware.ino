/*
 * ESP32-CAM firmware for stereo robot v2.
 *
 * Same as the original firmware but adds:
 *   /stream   — MJPEG stream (unchanged)
 *   /capture  — returns a single JPEG frame with headers:
 *                 X-Frame-Seq:  monotonic frame counter
 *                 X-Timestamp:  millis() at capture time
 *               The Python app uses /capture for synchronized grabs
 *               and /stream as a fallback.
 *
 * Flash both ESP32-CAMs with the same firmware.
 * Just change the WiFi credentials below.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ===== CHANGE THESE =====
const char* ssid     = "covalitos";
const char* password = "0526461671";
// =========================

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define PART_BOUNDARY "123456789000000000000987654321"

static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

volatile uint32_t frame_seq = 0;
httpd_handle_t stream_httpd = NULL;
httpd_handle_t capture_httpd = NULL;

// ── /stream handler (unchanged from original) ────
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[64];

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) { res = ESP_FAIL; break; }
        frame_seq++;

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res != ESP_OK) break;

        size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res != ESP_OK) break;

        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        esp_camera_fb_return(fb);
        fb = NULL;
        if (res != ESP_OK) break;
    }
    if (fb) esp_camera_fb_return(fb);
    return res;
}

// ── /capture handler (single frame with metadata) ─
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    uint32_t seq = ++frame_seq;
    uint32_t ts  = (uint32_t)millis();

    httpd_resp_set_type(req, "image/jpeg");

    char hdr[48];
    snprintf(hdr, sizeof(hdr), "%u", seq);
    httpd_resp_set_hdr(req, "X-Frame-Seq", hdr);

    // Use a second static buffer for the timestamp header
    static char ts_buf[48];
    snprintf(ts_buf, sizeof(ts_buf), "%u", ts);
    httpd_resp_set_hdr(req, "X-Timestamp", ts_buf);

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

void startServers() {
    // Stream server on port 80
    httpd_config_t cfg80 = HTTPD_DEFAULT_CONFIG();
    cfg80.server_port = 80;
    httpd_uri_t stream_uri = {
        .uri = "/stream", .method = HTTP_GET,
        .handler = stream_handler, .user_ctx = NULL
    };
    if (httpd_start(&stream_httpd, &cfg80) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    }

    // Capture server on port 81 (separate so /stream doesn't block /capture)
    httpd_config_t cfg81 = HTTPD_DEFAULT_CONFIG();
    cfg81.server_port = 81;
    httpd_uri_t capture_uri = {
        .uri = "/capture", .method = HTTP_GET,
        .handler = capture_handler, .user_ctx = NULL
    };
    if (httpd_start(&capture_httpd, &cfg81) == ESP_OK) {
        httpd_register_uri_handler(capture_httpd, &capture_uri);
    }
}

void setup() {
    Serial.begin(115200);

    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound()) {
        config.frame_size   = FRAMESIZE_VGA;
        config.jpeg_quality = 10;
        config.fb_count     = 2;
    } else {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 20;
        config.fb_count     = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }

    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
    }
    Serial.println();
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.println("  /stream  on port 80");
    Serial.println("  /capture on port 81");

    startServers();
}

void loop() {
    delay(1);
}
