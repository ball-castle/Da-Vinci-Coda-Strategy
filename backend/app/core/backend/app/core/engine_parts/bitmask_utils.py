# 位掩码核心工具与常量定义

# 达芬奇密码每个颜色有 12 张牌 (0-11)
# 我们使用低 12 位代表数字 0-11 的可用状态 (1为可用，0为排除)
FULL_MASK = 0b111111111111  # 4095

def create_mask(available_numbers: list[int]) -> int:
    \"\"\"将由整数组成的可用列表转换为一个位掩码\"\"\"
    mask = 0
    for num in available_numbers:
        if 0 <= num <= 11:
            mask |= (1 << num)
    return mask

def get_numbers_from_mask(mask: int) -> list[int]:
    \"\"\"从位掩码还原出所有可用的数字列表\"\"\"
    return [i for i in range(12) if (mask & (1 << i))]

def bit_count(mask: int) -> int:
    \"\"\"计算掩码中包含的可能组合数(相当于二进制中 1 的个数)\"\"\"
    return bin(mask).count('1')
