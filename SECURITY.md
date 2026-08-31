# Security policy

이 repository에는 API key, access token, password, private key, `.env` 파일 또는 사용자 홈 절대 경로를 commit하지 않습니다. 학습에 필요한 자격증명은 환경변수나 GitHub Actions secrets로만 전달해야 합니다.

보안 문제나 노출된 credential을 발견하면 public issue를 만들지 말고 GitHub의 **Report a vulnerability** 기능으로 비공개 제보해 주세요. 실제 credential이 노출된 경우에는 파일 삭제만으로 충분하지 않으므로 먼저 발급처에서 폐기·재발급한 뒤 Git history 정리를 진행해야 합니다.

PyTorch checkpoint는 신뢰할 수 있는 출처의 파일만 사용하고, 가능한 경우 `torch.load(..., weights_only=True)`로 불러오며 각 release의 `checkpoint_audit.json` SHA-256을 확인해 주세요.
