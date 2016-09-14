from getch import getch
z = getch()
chars = [9, 27]

while ord(z) not in chars:
  z = getch()
  print ord(z)

if ord(z) == 33:
  print 'a'
if ord(z) == 27:
  print 'b'
