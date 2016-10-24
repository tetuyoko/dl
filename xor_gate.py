import and_gate as ag
import nand_gate as nag
import or_gate as og

def XOR(x1, x2):
    s1 = nag.NAND(x1, x2)
    s2 = og.OR(x1, x2)
    y = ag.AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
