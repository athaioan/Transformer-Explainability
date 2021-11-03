import torch

def safe_divide(a, b):
    den1 = b.clamp(min=1e-9)
    den2 = b.clamp(max=1e-9)
    den = den1 + den2
    den = den + den.eq(0).type(den.type()) * 1e-9
    result = a / den * b.ne(0).type(b.type())
    return result

def trial(a, b):
    den = b
    den = den + den.eq(0).type(den.type()) * 1e-9
    division = a / den
    result = division * b.ne(0).type(b.type())
    return result

a = [1, 2, 3, 4, 5]
a = torch.FloatTensor(a)

b = [1, 2, 3, 0, 0]
b = torch.FloatTensor(b)

c = torch.divide(a, b)
d = safe_divide(a, b)
e = trial(a, b)

print(torch.equal(c, d))
print(torch.equal(c, e))