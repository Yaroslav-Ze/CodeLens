package demo.library;

public class LoanService {
    public void borrowBook(Book book, String readerId) {
        if (!book.isAvailable()) {
            throw new IllegalStateException("Book is not available");
        }
        book.markBorrowed();
        recordLoan(readerId);
    }

    public void returnBook(Book book) {
        book.markReturned();
    }

    private void recordLoan(String readerId) {
        System.out.println("Loan created for reader: " + readerId);
    }
}
