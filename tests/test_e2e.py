"""e2e：真伺服器 + 真瀏覽器 + 真模型，走完整使用者流程。"""
import socket
import subprocess
import time

import pytest
from playwright.sync_api import expect

PORT = 8321
URL = f"http://127.0.0.1:{PORT}"
GPT2 = "openai-community/gpt2"
GRANITE = "ibm-granite/granite-3.1-1b-a400m-instruct"


@pytest.fixture(scope="session")
def server():
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "server.main:app", "--port", str(PORT)])
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", PORT), 1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.fail("伺服器啟動逾時")
        yield URL
    finally:
        proc.terminate()
        proc.wait()


@pytest.mark.e2e
def test_full_user_flow(server, page):
    page.goto(server)
    page.select_option("#dense-select", GPT2)
    page.select_option("#moe-select", GRANITE)
    # 真實使用者行為：連點兩顆載入鈕再各自等待完成。server/models.py 已用
    # _LOAD_LOCK 序列化實際載入（避免 transformers 首次 import 的 lazy-module
    # 競態，以及並行 from_pretrained 踩踏全域預設 dtype），此處直接驗證修復。
    page.click("#btn-load-dense")
    page.click("#btn-load-moe")
    expect(page.locator("#dense-status")).to_have_text("✓ 已載入", timeout=600_000)
    expect(page.locator("#moe-status")).to_have_text("✓ 已載入", timeout=600_000)

    page.fill("#sentence", "今天天氣真好")
    expect(page.locator("#btn-run")).to_be_enabled(timeout=10_000)
    page.click("#btn-run")

    expect(page.locator("#dense-panel rect").first).to_be_visible(timeout=120_000)
    expect(page.locator("#moe-panel rect").first).to_be_visible()
    expect(page.locator("#heatmap rect").first).to_be_visible()
    page.wait_for_timeout(3000)  # 播放中
    assert page.text_content("#dense-counter") != "0"
    assert page.text_content("#moe-counter") != "0"
    assert (page.locator("#dense-tokens span").count() > 0
            and page.locator("#moe-tokens span").count() > 0)
