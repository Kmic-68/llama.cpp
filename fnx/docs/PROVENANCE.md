# Provenance of every file in `engine/`

Compared with Strata v0.2.10 (`github.com/Niko1221/Strata`, commit `ffa80d73133ea5c548aefe57889a16e8085a938f`), directory `src/split/`.
Strata's `src/split/` is itself a port of sFNX by John Fink (`third_party/sfnx/NOTICE.md`). The full diff is `upstream-v0.2.10.diff`.
SHA-256 values are those of the sources of our Edge0 production engine
(`strata-split-2`, binary SHA-256 `f6088fdee182bbe6…`, 15 Nov 2026) with 18 Nov's patch applied (see the end of this
paragraph). That engine's sources are our Coder engine (`40ef6d531143d73d…`: our UD-Q4_K_XL production engine of 13 Nov,
`c213636ce3de2182…`, with `history/patches/maska_patch.py` applied) with the five `history/patches/edge0_patch*.py` applied.
With `FNX_EXPERT_MASK` and `FNX_LATENT` unset it gives our production engines' bits (see GUIDE.md, "Our numbers", "Coder" and
"Edge0-145B"). Since 16 Nov 2026 our UD-Q4_K_XL production engine is `6ccc87af7e9baf50…`: `c213636ce3de2182…`'s sources
built with `-DSTRATA_SPLIT_PF_JOB=96` (no source file differs; GUIDE.md, "Build"). This folder's sources of then built with that
option (`e56c3f604d2eb5a7…`) gave its first-token logits in 29 of 29 requests of the benchmark, and `PF_JOB=96 bash build.sh`'s
own build on our server (`a1e9ac07946cee2f…`) gave the same ones. Since 17 Nov 2026 `build.sh` builds with that option by
default, `e56c3f604d2eb5a7…` is our Edge0 production engine, and our IQ3_S production engine is `36409ddc524cd93e…`:
`bf9c7f1f22b0af4a…`'s sources built with that option (the same first-token logits in 27 of 27 requests of our test chain).
Since 18 Nov 2026 `src/engine.cu`, `src/kernels.cu`, `src/kernels.hpp` and `src/prefill.cu` also have
`history/patches/mtp_norm_patch.py` applied (`FNX_MTP_NORM_STREAMS`; unset, the code makes the calls it made before). The
sources as they are here were built with that option on 17 Nov (`a77172354b195150…`): that engine is in none of our
profiles and has not run in a round or in the benchmark, and `build.sh` has not been run on them. Our UD-Q4_K_XL
production engine since 18 Nov is `f28228d145e981b5…`: `6ccc87af7e9baf50…`'s sources with the same patch, built with that
option (GUIDE.md, "Unsloth's UD-Q4_K_XL" and "Known limitations").

| File | Origin | SHA-256 |
|---|---|---|
| `CMakeLists.txt` | Strata v0.2.10, modified by us | `f6e24d0eb63c32140a383e524a81f355b6b828a5cc805d1c427b9180bb75b505` |
| `data/coder_mask.txt` | ours (data): the experts ISTA-DASLab's Coder release keeps (Apache-2.0), from its allocation list by scripts/coder_mask.py | `a38e41c274adb1421adc3589806aff96c7ff1ab4c57ed5e214cb06561e51937d` |
| `data/draft_vocab_en.bin` | Strata v0.2.10, unchanged (from sFNX) | `369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25` |
| `data/draft_vocab_sl.bin` | ours (data): Strata's tools/draft_vocab.py over Slovenian Wikipedia, base data/draft_vocab_en.bin | `3e3fa1a67d8dbe0989ff682758eb9467ee827174ab3471086b6794b81e50620b` |
| `data/expert-routes.bin` | Strata v0.2.10, unchanged (from sFNX) | `8d6911494194efb24f0627001d33304a1e6664d5ae6363b09ccb025fc6f52930` |
| `p100_compat.h` | ours (new) | `0bda7379f15903a165c008c19456380cf0b85e41f486079ef877a3d65f34bcac` |
| `src/attn_prompt.cuh` | Strata v0.2.10, modified by us | `5ecdc42c781d9eb7e2c925ed2980d3d4f16800728708c35189eb3f68840e2987` |
| `src/batch.cpp` | Strata v0.2.10, modified by us | `22635af85694cb533013c74aa5f744c99d8779e75d34633e9ddbe5344ef96b07` |
| `src/batch.hpp` | Strata v0.2.10, unchanged | `24af1846f0acc1770ebb87f6b9e602d85f6d5ba3bdc2b9b402c076bc9c614a96` |
| `src/batch_plan.hpp` | Strata v0.2.10, unchanged | `16c6d6d7ac5bac8e48a80a202541dde6b2b00b95b64abf168815dfb13e05a831` |
| `src/card_check.hpp` | Strata v0.2.10, unchanged | `704aeda84671d56f6021321ee04562562ffa2358661b73312399184396558320` |
| `src/card_split.hpp` | Strata v0.2.10, unchanged | `0e211d82a6b0560078e9b5a6e2e681394b412f5344c08b92f877dc9fc0b0e2cf` |
| `src/dense.cuh` | Strata v0.2.10, modified by us | `1ca66d9ac98ea99c5f7dc388142cb04396440b2a8be38b2dd10e34ee4e2b62ba` |
| `src/engine.cu` | Strata v0.2.10, modified by us | `631f270c9d83e7f6aa95036ec6264ebe7d075ff546073c656fcaafc3cd94c0ca` |
| `src/engine.hpp` | Strata v0.2.10, modified by us | `7c42647a14be2e63f20f7c6a56ab051d1dacf2a7e7da7cf8cab8f2f9fd4fe5c4` |
| `src/engine_common.cu` | Strata v0.2.10, modified by us | `7787e092a92f2330fd08ccca8b6302f6ad65e2472c93de9aa5dead9f625b8bd9` |
| `src/engine_impl.hpp` | Strata v0.2.10, modified by us | `97a32711f217a6c06bc929d8fc3024f2aa0c52df61de6d03144942731e212fd1` |
| `src/engine_parts.hpp` | Strata v0.2.10, modified by us | `4af1fd9aedf96f241d1f202ea924ea722adb63ea77942b5b1a658e6e2ffeba1e` |
| `src/env.hpp` | Strata v0.2.10, unchanged | `4dc19a4a18ddc95fb1a3e1a60fb70821d78b459de626d70385e8339a402aa134` |
| `src/gdn_core.cuh` | Strata v0.2.10, modified by us | `fd373392625b8f293722c2014ba4f337db0cd9dcdb5b56398d22ae21de5c83df` |
| `src/gdn_prompt.cuh` | Strata v0.2.10, unchanged | `07e5e0588aa96bb18ac639007b53d5fe5b8641f6adb552ed77501296f9271387` |
| `src/gguf.cpp` | Strata v0.2.10, modified by us | `bd6339d95de593470535eea838bfebed0a3913b4ee4bd55fbdb151e3236a2f50` |
| `src/gguf.hpp` | Strata v0.2.10, modified by us | `c8a93acebad4dd383f31895d05e27e3db1424eaa78e79fd6a33e80e4c6c15bf0` |
| `src/hc.cuh` | Strata v0.2.10, modified by us | `27a9690e3521332a0db39f01d7e82aa0fb6fb32153cd96d0b483d7a334d1f54d` |
| `src/hip/cuda.h` | Strata v0.2.10, unchanged | `ef6d41d107c2c3a5ba78b01fa2ab6e5463099d14e3e7f507eae37a189e6a048e` |
| `src/hip/fnx_hip.h` | Strata v0.2.10, unchanged | `b8aa61996861f38912cfc09a0b960720fed05627d6d0ed91f0f243a8eb77a480` |
| `src/hip/mma.h` | Strata v0.2.10, unchanged | `ef5f3e18d7c801397427a4110222ce0494680971ff7bb514f8c0d78aa073f3aa` |
| `src/hip/peer.h` | Strata v0.2.10, unchanged | `c8062be7e8e1fb84dc20038cef61cc57d7eeeadab3c1a9944e61c3101b64c039` |
| `src/hip/peer_policy.hpp` | Strata v0.2.10, unchanged | `2d1379c022c2abba75fda1145bc79868adc8e0cab713960184987d0a9a3045bd` |
| `src/hip/rccl.h` | Strata v0.2.10, unchanged | `3869b1680e0377bbdf74272c95a2b8d8fceccd3df4d73f458ce1e30b27ff0aef` |
| `src/hip/rccl_dl.hpp` | Strata v0.2.10, unchanged | `e7d9ea5291dd907900f81c9521f64dad9d94770d023513f9b2ac6ef78b805167` |
| `src/hip/selftest_policy.hpp` | Strata v0.2.10, unchanged | `3ba715a7916f8900e76fc9b9e5464cdea053b51be1d0c28a8d68206dd7482cf9` |
| `src/iq3s_x.cuh` | ours (new) | `b3e208682ef2a76e78c9b6154e792dc1f39129f8fd7de381bac8a6fd6568146a` |
| `src/kernels.cu` | Strata v0.2.10, modified by us | `002f4ac82df95970c3e05e821f5b4cad3effd599c021a85d8d1afed4ae2774c7` |
| `src/kernels.hpp` | Strata v0.2.10, modified by us | `e9cbb275c2c9f28a431ce8aae4bd43ab7de2a4d2549bfbbb82639907692bbd8e` |
| `src/model.cpp` | Strata v0.2.10, modified by us | `3fdb890d2a67771a5132ef46686dbec5e1323b17bb8a2372372cfba56f25a4b9` |
| `src/model.hpp` | Strata v0.2.10, modified by us | `51830eb21547acbac1bb0d0279ec99f774092c82723b32ab2b46eb7ac3220c7a` |
| `src/moe.cuh` | Strata v0.2.10, modified by us | `cfdffc68ff915639d7ef036d7e35bd67320c37e44511cc0964d10671f46598ac` |
| `src/moe_mma.cu` | Strata v0.2.10, modified by us | `c66a70054fee82feca74ecf594eff5095dc207d84cee478e373f83bc92079b92` |
| `src/moe_mma.cuh` | Strata v0.2.10, unchanged | `67d3739844a7615dac39c3f6e4f034ef8818ee4898764df813a53b69003ab9a3` |
| `src/moe_mma_iq.cuh` | Strata v0.2.10, unchanged | `a59ca799ba878a7d9e6efc387d0cc55e3f7d274f75ec728d1e81acef6970d0b9` |
| `src/moe_q4k.cuh` | Strata v0.2.10, unchanged | `07d507e17b65ca32605babf61d60b83c6c3e02b5f4d9b3a75278b7f699ef8acc` |
| `src/moe_w.cuh` | Strata v0.2.10, unchanged | `c1562b445cc670d3cc6feebee4b853a523c41c5fbf797396caa88e129cb87714` |
| `src/owned.hpp` | Strata v0.2.10, unchanged | `bee34d928b503501015c207e10200a687d6b658961bf31a4e398435672f4cd82` |
| `src/p100_attn.cuh` | ours (new) | `4c12f01dde718b5522d474a341c395e0e0afb4963994da096aad48a193e4a3cc` |
| `src/p100_dma.cuh` | ours (new) | `3fb9504019588f25eb09be21f87c9962b44d898489d468fd0916c0c592725b8a` |
| `src/p100_dma_box.cuh` | ours (new) | `3eb3e1c8de418cbd27705577f761f783334ac72b411c27ebb4e25c39e6ff16f6` |
| `src/p100_iq_gemm.cuh` | ours (new) | `f0c6745211d9b423cd745095ff685da32bd983f62c03c1a5e6a5ea19bb81f45e` |
| `src/p100_moe.cu` | ours (new) | `7ffea370d816fc13d34f029f85bfca068ea2c6280ba2a9a212e7d04181f736e9` |
| `src/p100_moe_rows.cuh` | ours (new) | `7a3eb105eea9f7edcc666b1be859d1293660f6c9f2386989721c08568f7ae3ce` |
| `src/p100_over.hpp` | ours (new) | `ee4decf5138ff656fdfea8dfd6059d684871e3ae2c153077b1c88586f0a93f15` |
| `src/peer_probe.hpp` | Strata v0.2.10, unchanged | `e5bc045c8f97d25b4cfc5c3b3d0e7b1f7d0cc258a6756ea13ceb7283fe2a0554` |
| `src/pf_common.cuh` | Strata v0.2.10, modified by us | `5515e525fd5036bfecf7facb806b1622a89df0a183a0d1dc58d4b0be5328c116` |
| `src/pf_ring_plan.hpp` | Strata v0.2.10, unchanged | `ce52cb8fc7812a9ecd31742d0c3c35999a62e1c6d3cb8d2f7ba971806aff59aa` |
| `src/prefill.cu` | Strata v0.2.10, modified by us | `8618c166f6c746b7eab2fabfe12f26140883cb226a75cbb13437a7544453be3b` |
| `src/qsa.cuh` | Strata v0.2.10, modified by us | `7e94bdde945da43bb627c9bf7ed6fa8ff37a836741db8ebf1b5119dc6bba2d51` |
| `src/rope.hpp` | Strata v0.2.10, unchanged | `91c54cfda3b34a0f4c515b35c11a57d2b76a6662e350b86a71dd73bd455568a1` |
| `src/rope_gguf.hpp` | Strata v0.2.10, unchanged | `72ca61924332a6ef312f0a2ddf2fa6ace8368aee02e892fdb66333aebac19e56` |
| `src/selftest.cu` | Strata v0.2.10, unchanged | `3e73d783a2775ac2e8b9423147063ca9ae2103c046b8e77e687d2aa08f75a26a` |
| `src/selftest.hpp` | Strata v0.2.10, unchanged | `1ffd5c79ba54ff6b9dd1c636e160143d847aa5fa140e9bcc19e0c4cf41de779f` |
| `src/session.cpp` | Strata v0.2.10, modified by us | `349cfda640b2662ffc0a1e80800257a49806b09fc7878841557f95330709e786` |
| `src/session.hpp` | Strata v0.2.10, modified by us | `b487b6f27c65ddbb963d33209d0dd8690e3638c146d4f0e1376993ca4d09d88e` |
| `src/store.cpp` | Strata v0.2.10, unchanged | `b4685346d746ecb28867b1aab3d0e7e2490a1a748770e2367f027afdb9e0f1f2` |
| `src/store.hpp` | Strata v0.2.10, unchanged | `0cfedc0be22cf71414722087dacbb403be0c0beb05498121c7fa39123d501edb` |
| `src/tensor_convert.hpp` | Strata v0.2.10, modified by us | `bc3926d1426c2c2c2a14e47e27a6d0a27ee40ddf5676596885751c3339594cb8` |
| `src/topk.cuh` | Strata v0.2.10, modified by us | `a1702ff1d335f66dc729512c55ca72cf506806b07e8d13411eb8abc8912d4d67` |
| `tools/batch_check.hpp` | Strata v0.2.10, unchanged | `f311ba413ff21ced83610a6f7396ddf025a2ae35c437841a19b258c23e80d8bc` |
| `tools/batch_plan_test.cpp` | Strata v0.2.10, unchanged | `49245a16655388fe553d3c0ed2f354c5c62215379ec2443cd0b54fe29655842b` |
| `tools/card_check_test.cpp` | Strata v0.2.10, unchanged | `b4133d30288f4c58bded6db96e5120e102888c37ebbc4c1586b7709322909e5b` |
| `tools/convert_dump.cpp` | Strata v0.2.10, unchanged | `8675163024ce5cd027422c7e2b099be68333b4d31f4b4d50dfeecf5914c6ea36` |
| `tools/fnx_run.cpp` | Strata v0.2.10, unchanged | `bda98702e4d774e1e58991c16674b306baceb51cfcbfdbd670fdb5d1fbc1b48e` |
| `tools/gguf_test.cpp` | Strata v0.2.10, unchanged | `025c036e10e174d35f58b49c9af6320c660e6dbbb20b9819e73cf68c6a23d51d` |
| `tools/moe_tile_test.cu` | Strata v0.2.10, unchanged | `92ba9ee97f80afb9cfc62f2795f26ba30e13f1345444457313c8d803f35fc3ac` |
| `tools/peer_policy_test.cpp` | Strata v0.2.10, unchanged | `8cd881d7d2978359847f51404235baea3ef6d9afd1edc9a5692119932854c64e` |
| `tools/pf_ring_plan_test.cpp` | Strata v0.2.10, unchanged | `3e45833ae8b1d73ed35ea1c490c73fef8cfaf828c3c117ff9feaae5f6382f4af` |
| `tools/rccl_policy_test.cpp` | Strata v0.2.10, unchanged | `6ad201a68854d6933c776eb5f4f27f5dcb8334cbcddb3a12d18bf34e46e29042` |
| `tools/ref_tokens.hpp` | Strata v0.2.10, unchanged | `52f624c13fe36c24b04ebc0e9c7b137426ebd691b2a4cc5b64cd36dcb5722664` |
| `tools/rope_test.cpp` | Strata v0.2.10, unchanged | `1d15c56b928fe31f07cd82e65a4a567542d5f6253fb4e88244fb8dc8665888c1` |
| `tools/selftest_main.cpp` | Strata v0.2.10, unchanged | `c865567c1f720d2bb9cfbcae6ee711a65b760fdfe6075a592b589b9197d8a7a9` |
| `tools/selftest_policy_test.cpp` | Strata v0.2.10, unchanged | `a7efa0b1e19f78e90e502989a35e4dd2b68e9e4592339b465e13fb4f9262f1c3` |
| `tools/sfnx.cpp` | Strata v0.2.10, modified by us | `115a2439cde48ab9df6dc7dac7900a442811f45250c463a865151cccd2a6c90f` |
| `tools/sfnx_input.hpp` | Strata v0.2.10, unchanged | `bbd3cbfc7369eae44ee85f8d10dcc421eaa549250a4e95cc125f77c2467eff69` |
| `tools/sfnx_input_test.cpp` | Strata v0.2.10, unchanged | `432c1f4ff0c1bef3a40999baa8d6ec58f70f0561b734ae486134db7f4b34d809` |

Totals: Strata v0.2.10, modified by us: 27; Strata v0.2.10, unchanged: 43; Strata v0.2.10, unchanged (from sFNX): 2; ours: 11
