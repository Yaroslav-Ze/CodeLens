package demo.library;

import java.util.ArrayList;
import java.util.List;

public class CatalogService {
    private final List<Book> books = new ArrayList<>();

    public void addBook(Book book) {
        books.add(book);
    }

    public List<Book> findAvailableBooksByAuthor(String author) {
        return books.stream()
                .filter(Book::isAvailable)
                .filter(book -> book.getAuthor().equalsIgnoreCase(author))
                .toList();
    }
}
