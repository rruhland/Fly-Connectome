This experimental dependency is OgmaNeo2 at c5be9ec41d26aeeb2fc805961140781c851542cf.
https://github.com/ogmacorp/OgmaNeo2/tree/c5be9ec41d26aeeb2fc805961140781c851542cf
It retains the included CC BY-NC-SA 4.0 license and Ogma attribution; it is not
relicensed under the surrounding project's license. Adaptations in Actor.h/.cpp:
zero-initialized local action/value weights; fixed prior logits; saved actual
behavior probabilities; per-transition elapsed discount; terminal bootstrap zero;
zero-TD policy update zero; extra replay/audit accessors and state serialization.
Hierarchy gains a complete runtime reset that retains learned weights.
No error is propagated between learned layers. The delayed signed-TD policy
updates are a heuristic actor-critic; replayed gradients are not claimed unbiased.
