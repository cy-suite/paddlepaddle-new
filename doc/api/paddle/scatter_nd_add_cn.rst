                    output = [0, 22, 12, 14, 4, 5]

**示例一图解说明**：

    在这个示例中，通过 Paddle 的 scatter_nd_add 函数对张量 x 进行稀疏加法操作。初始张量 x 为 [0, 1, 2, 3, 4, 5]，通过 index 指定需要更新的索引位置，并使用 updates 中的值进行累加。scatter_nd_add 函数会根据 index 的位置逐步累加 updates 中的对应值，而不是替换原有值，最终得到输出张量为 [0, 22, 12, 14, 4, 5]，实现了对张量部分元素的累加更新而保持其他元素不变。
    .. figure:: ../../images/api_legend/scatter_nd_add.png
       :width: 700
       :alt: 示例一图示
       :align: center
