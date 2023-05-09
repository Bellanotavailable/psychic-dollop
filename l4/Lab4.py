class Polynomial:
    def __init__(self, coef: list):
        self._coef_ = tuple(map(float, coef))
        try:
            while self._coef_[0] == 0:
                self._coef_ = self._coef_[1:]
        except IndexError:
            self._coef_ = (0,)

    def __str__(self):
        res = ''
        if self.degree() > 1:
            el = self._coef_[0]
            if abs(el) != 1:
                res += str(el)
            elif el == -1:
                res += '-'
            res += 'x**' + str(self.degree())

            for i, el in enumerate(self._coef_[1:-2], start=1):
                if el == 0:
                    continue
                elif el > 0:
                    res += ' + '
                else:
                    res += ' - '

                if abs(el) != 1:
                    res += str(abs(el))

                res += 'x**' + str(self.degree() - i)

            el = self._coef_[-2]
            if el != 0:
                if el > 0:
                    res += ' + '
                else:
                    res += ' - '

                if abs(el) != 1:
                    res += str(abs(el))
                res += 'x'

            el = self._coef_[-1]
            if el != 0:
                if el > 0:
                    res += ' + '
                else:
                    res += ' - '
                res += str(abs(el))
        elif self.degree() == 1:
            el = self._coef_[0]
            if el == 1:
                res += 'x'
            elif el == -1:
                res += '-x'
            else:
                res += str(el) + 'x'

            el = self._coef_[-1]
            if el != 0:
                if el > 0:
                    res += ' + '
                else:
                    res += ' - '
                res += str(abs(el))
        else:
            res = str(self._coef_[0])
        return res

    def degree(self):
        return len(self._coef_) - 1

    def coef(self):
        return self._coef_

    def value(self, point):
        # print("---->>> ", self._coef_)
        # sum = self._coef_[0] * point + self._coef_[1]
        # if (len(self._coef_) > 2):
        #     for i in range(2, len(self._coef_)):
        #         sum += sum * point + self._coef_[i]

        # return (sum)
        return sum([el * point ** i for i, el in enumerate(self._coef_[::-1])])

    def sum(self, other):
        other_coef = list(getattr(other, '_coef_'))
        delta = len(self._coef_) - len(other_coef)

        # self is longer
        if delta >= 0:
            res = list(self._coef_)[::-1]
            for i, c in enumerate(other_coef[::-1]):
                res[i] += c
        # other is longer
        else:
            res = other_coef[::-1]
            for i, c in enumerate(self._coef_[::-1]):
                res[i] += c

        return Polynomial(res[::-1])

    def difference(self, other):
        other_coef = list(getattr(other, '_coef_'))
        delta = len(self._coef_) - len(other_coef)

        # self is longer
        if delta >= 0:
            res = list(self._coef_)[::-1]
            for i, c in enumerate(other_coef[::-1]):
                res[i] -= c
        # other is longer
        else:
            res = list(map(lambda x: -x, other_coef[::-1]))
            for i, c in enumerate(self._coef_[::-1]):
                res[i] += c

        return Polynomial(res[::-1])

    def __mult_by_simple__(self, coef, degree):
        return Polynomial(list(map(lambda x: coef * x, self._coef_)) + [0.0] * degree)

    def prod(self, other):
        n = other.degree()
        pre_res = [self.__mult_by_simple__(c, n - i) for i, c in enumerate(other.coef())]
        res = pre_res[0]
        for pol in pre_res[1:]:
            res = res.sum(pol)
        return res

    def pov(self, n):
        n = int(n)
        if n < 0:
            raise AssertionError('Power cannot be negative')
        if n == 0:
            return Polynomial([1])
        res = self
        for i in range(n - 1):
            res = res.prod(self)
        return res

    def division(self, other):
        if other.coef() == [0]:
            raise ZeroDivisionError('Polynomial is zeros')

        if other.degree() == 0:
            return self.__mult_by_simple__(1 / other.coef()[0], 0), Polynomial([0])

        dividend = self
        divisor = other
        res = Polynomial([0])
        while dividend.degree() >= divisor.degree():
            deg = dividend.degree() - divisor.degree()
            coef = dividend.coef()[0] / divisor.coef()[0]

            res = res.sum(Polynomial([coef] + [0.0] * deg))
            dividend = dividend.difference(divisor.__mult_by_simple__(coef, deg))

        return res, dividend

    def diff(self, n=1):
        n = int(n)
        if n < 1:
            raise AssertionError('The order of the derivative cannot be less than one')

        res = [el * i for i, el in enumerate(self._coef_[-2::-1], start=1)]
        n -= 1
        if n == 0:
            return Polynomial(res[::-1])
        else:
            return Polynomial(res[::-1]).diff(n)

    def integrate(self, left, right):
        res = Polynomial([el / i for i, el in enumerate(self._coef_[::-1], start=1)][::-1] + [0.0])
        return res.value(right) - res.value(left)


if __name__ == '__main__':
    obj1 = Polynomial([2, -6, 5, 15])
    obj2 = Polynomial([2, 5, 5])
    print(obj1)
    print(obj2, end='\n\n')
    print(f'SUM:\nfirst way {obj1.sum(obj2)}\nsecond way {obj2.sum(obj1)}\n')
    print(f'DIFFERENCE:\nfirst way {obj1.difference(obj2)}\nsecond way {obj2.difference(obj1)}\n')
    print(f'PRODUCTION\nfirst way {obj1.prod(obj2)}\nsecond way {obj2.prod(obj1)}\n')
    print(f'POWER\nobj1 ** 0: {obj1.pov(0)}\nobj2 ** 3: {obj2.pov(3)}\n')

    div1, mod1 = obj1.division(obj2)
    div2, mod2 = obj2.division(obj1)
    print(f'DIVISION\nobj1/obj2 div = {div1}, mod = {mod1}\nobj2/obj1 div = {div2}, mod = {mod2}\n')

    print('DIFF')
    for i in range(1, 6):
        print(f'for obj1 {i}: {obj1.diff(i)}')

    print('\nINTEGRAL')
    print(f'integrate obj1 from -1 to 1: {obj1.integrate(-1,1)}\nintegrate obj2 from 5 to 0: {obj2.integrate(5, 0)}')