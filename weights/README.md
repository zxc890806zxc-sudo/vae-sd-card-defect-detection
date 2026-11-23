# 權重檔 / Model weights

權重檔太大（每個約 662 MB），沒有放在 repo 裡，請到本專案的 [Releases](https://github.com/zxc890806zxc-sudo/vae-sd-card-defect-detection/releases) 頁面下載。

Weight files are too large for the repository. Download them from the [Releases](https://github.com/zxc890806zxc-sudo/vae-sd-card-defect-detection/releases) page.

下載後放到要執行的程式**同一個資料夾**，例如 `src/single_card/vae_model.weights.h5`。程式用相對路徑讀取，檔名不能更改。

| 檔案 / File | 使用的程式 / Used by | 大小 / Size | SHA-256 |
| --- | --- | ---: | --- |
| `vae_model.weights.h5` | `src/single_card/` | 661.69 MiB | `0C8EBB9DEBC0D21D0CFF88BE432720D307E08F7ADBCBACFECA4C3BD130029BD2` |
| `vae_D1.weights.h5` | `src/multi_region/`, `src/multi_region_camera_v2/`（區域 D1） | 661.69 MiB | `376A6859963C5DC7DEEACFFF3BF5E24F303E60592F8A2709ACE6C50DAA97FB54` |
| `vae_D2.weights.h5` | 同上（區域 D2） | 661.69 MiB | `E7DA2B9D3BFF2249E0DCF29144A72BE465CAC52AFDA7A051C37F15FA2AAD35B8` |
| `vae_D3.weights.h5` | 同上（區域 D3） | 661.69 MiB | `F181B2E2F71717E0648455A7F4D61FE77DEF2BDC8895D6F37104991F189E66D0` |
| `vae_D4.weights.h5` | 同上（區域 D4） | 661.69 MiB | `60AD6C5D80406A289C4DA3FBE932D71A149697C0488E812D29100D948E48A959` |

在 PowerShell 檢查下載的檔案是否完整：

```powershell
Get-FileHash .\vae_model.weights.h5 -Algorithm SHA256
```
